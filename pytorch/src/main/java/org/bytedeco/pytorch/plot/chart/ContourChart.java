package org.bytedeco.pytorch.plot.chart;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Contour line chart (matplotlib {@code plt.contour}).
 * Accepts Z matrix (row-major) with optional X/Y coordinate vectors,
 * or full meshgrid X,Y,Z (uses first row of X and first col of Y when 2D).
 *
 * <p>Approximate marching-squares style polylines — not bit-identical to MPL.
 */
public final class ContourChart extends BaseChart {
    private final double[][] z;
    private final double[] xCoords;
    private final double[] yCoords;
    private int nLevels = 8;
    private String cmap = "coolwarm";
    private boolean showColorbar = true;
    private double[] explicitLevels;

    public ContourChart(String title, double[][] z) {
        this(title, null, null, z);
    }

    public ContourChart(String title, double[] x, double[] y, double[][] z) {
        super(title == null ? "contour" : title);
        this.z = z;
        int rows = z.length;
        int cols = rows == 0 ? 0 : z[0].length;
        if (x != null && x.length == cols) this.xCoords = x;
        else {
            this.xCoords = new double[cols];
            for (int i = 0; i < cols; i++) this.xCoords[i] = i;
        }
        if (y != null && y.length == rows) this.yCoords = y;
        else {
            this.yCoords = new double[rows];
            for (int i = 0; i < rows; i++) this.yCoords[i] = i;
        }
    }

    /** From meshgrid-style 2D X,Y,Z (uses X[0][*] and Y[*][0]). */
    public static ContourChart fromMesh(String title, double[][] X, double[][] Y, double[][] Z) {
        int rows = Z.length;
        int cols = rows == 0 ? 0 : Z[0].length;
        double[] x = new double[cols];
        double[] y = new double[rows];
        if (X != null && X.length > 0) {
            for (int j = 0; j < cols && j < X[0].length; j++) x[j] = X[0][j];
        } else {
            for (int j = 0; j < cols; j++) x[j] = j;
        }
        if (Y != null) {
            for (int i = 0; i < rows && i < Y.length; i++) y[i] = Y[i][0];
        } else {
            for (int i = 0; i < rows; i++) y[i] = i;
        }
        return new ContourChart(title, x, y, Z);
    }

    public ContourChart setLevels(int n) { this.nLevels = Math.max(1, n); return this; }
    public ContourChart setLevels(double[] levels) { this.explicitLevels = levels; return this; }
    public ContourChart setCmap(String name) {
        this.cmap = name == null ? "coolwarm" : name.toLowerCase(Locale.ROOT);
        return this;
    }
    public ContourChart setShowColorbar(boolean v) { this.showColorbar = v; return this; }

    @Override public ContourChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public ContourChart setSize(int w, int h) { super.setSize(w, h); return this; }

    @Override
    public BufferedImage render() {
        int rows = z.length;
        int cols = rows == 0 ? 0 : z[0].length;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int cbW = showColorbar ? 40 : 0;
        int left = 60, right = 20 + cbW, top = 40, bottom = 50;
        int plotW = width - left - right, plotH = height - top - bottom;

        double zMin = Double.POSITIVE_INFINITY, zMax = Double.NEGATIVE_INFINITY;
        for (double[] row : z) for (double v : row) {
            if (!Double.isNaN(v)) { zMin = Math.min(zMin, v); zMax = Math.max(zMax, v); }
        }
        if (zMax <= zMin) { zMin = 0; zMax = 1; }

        double xMin = min(xCoords), xMax = max(xCoords);
        double yMin = min(yCoords), yMax = max(yCoords);
        if (xMax <= xMin) { xMin -= 0.5; xMax += 0.5; }
        if (yMax <= yMin) { yMin -= 0.5; yMax += 0.5; }
        double[] xr = {xMin, xMax}, yr = {yMin, yMax};
        padRange(xr); padRange(yr);

        drawAxesFrame(g, left, top, plotW, plotH, xr[0], xr[1], yr[0], yr[1]);

        // filled background via bilinear-ish cell coloring (coarse)
        for (int i = 0; i < rows - 1; i++) {
            for (int j = 0; j < cols - 1; j++) {
                double v = (z[i][j] + z[i][j + 1] + z[i + 1][j] + z[i + 1][j + 1]) / 4.0;
                float t = (float) ((v - zMin) / (zMax - zMin));
                t = Math.max(0f, Math.min(1f, t));
                g.setColor(new Color(
                    HeatmapChart.mapColor(t, cmap).getRed(),
                    HeatmapChart.mapColor(t, cmap).getGreen(),
                    HeatmapChart.mapColor(t, cmap).getBlue(),
                    80));
                int x0 = mapX(xCoords[j], xr[0], xr[1], left, plotW);
                int x1 = mapX(xCoords[Math.min(j + 1, cols - 1)], xr[0], xr[1], left, plotW);
                int y0 = mapY(yCoords[i], yr[0], yr[1], top, plotH);
                int y1 = mapY(yCoords[Math.min(i + 1, rows - 1)], yr[0], yr[1], top, plotH);
                int px = Math.min(x0, x1), py = Math.min(y0, y1);
                g.fillRect(px, py, Math.max(1, Math.abs(x1 - x0)), Math.max(1, Math.abs(y1 - y0)));
            }
        }

        double[] levels = explicitLevels;
        if (levels == null) {
            levels = new double[nLevels];
            for (int i = 0; i < nLevels; i++)
                levels[i] = zMin + (zMax - zMin) * (i + 1) / (nLevels + 1.0);
        }

        g.setStroke(new BasicStroke(1.4f));
        for (int li = 0; li < levels.length; li++) {
            double level = levels[li];
            float t = (float) ((level - zMin) / (zMax - zMin));
            g.setColor(HeatmapChart.mapColor(Math.max(0f, Math.min(1f, t)), cmap));
            // marching squares on each cell
            for (int i = 0; i < rows - 1; i++) {
                for (int j = 0; j < cols - 1; j++) {
                    double v00 = z[i][j], v10 = z[i][j + 1];
                    double v01 = z[i + 1][j], v11 = z[i + 1][j + 1];
                    int code = 0;
                    if (v00 >= level) code |= 1;
                    if (v10 >= level) code |= 2;
                    if (v11 >= level) code |= 4;
                    if (v01 >= level) code |= 8;
                    if (code == 0 || code == 15) continue;
                    List<double[]> pts = edgePoints(code, level,
                        xCoords[j], xCoords[j + 1], yCoords[i], yCoords[i + 1],
                        v00, v10, v11, v01);
                    if (pts.size() >= 2) {
                        int x0 = mapX(pts.get(0)[0], xr[0], xr[1], left, plotW);
                        int y0 = mapY(pts.get(0)[1], yr[0], yr[1], top, plotH);
                        int x1 = mapX(pts.get(1)[0], xr[0], xr[1], left, plotW);
                        int y1 = mapY(pts.get(1)[1], yr[0], yr[1], top, plotH);
                        g.drawLine(x0, y0, x1, y1);
                    }
                }
            }
        }

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
            g.drawString(formatTick(zMax), cbX + 14, top + 10);
            g.drawString(formatTick(zMin), cbX + 14, top + plotH);
        }
        g.dispose();
        return img;
    }

    private static List<double[]> edgePoints(int code, double level,
            double x0, double x1, double y0, double y1,
            double v00, double v10, double v11, double v01) {
        List<double[]> pts = new ArrayList<>(2);
        // bottom edge (y0): v00--v10
        if (((code & 1) == 0) != ((code & 2) == 0))
            pts.add(new double[]{lerp(x0, x1, v00, v10, level), y0});
        // right edge (x1): v10--v11
        if (((code & 2) == 0) != ((code & 4) == 0))
            pts.add(new double[]{x1, lerp(y0, y1, v10, v11, level)});
        // top edge (y1): v01--v11
        if (((code & 8) == 0) != ((code & 4) == 0))
            pts.add(new double[]{lerp(x0, x1, v01, v11, level), y1});
        // left edge (x0): v00--v01
        if (((code & 1) == 0) != ((code & 8) == 0))
            pts.add(new double[]{x0, lerp(y0, y1, v00, v01, level)});
        // keep first two
        if (pts.size() > 2) return pts.subList(0, 2);
        return pts;
    }

    private static double lerp(double a, double b, double va, double vb, double level) {
        double den = vb - va;
        if (Math.abs(den) < 1e-15) return (a + b) / 2;
        double t = (level - va) / den;
        t = Math.max(0, Math.min(1, t));
        return a + t * (b - a);
    }
}
