package samples;

import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.utils.plot.BaseChart;
import org.bytedeco.pytorch.utils.plot.LineChart;
import org.bytedeco.pytorch.utils.plot.Matplotlib;
import org.bytedeco.pytorch.utils.plot.Seaborn;

/**
 * Interactive demo: each chart {@code show()} <b>blocks</b> until you close the
 * window — so figures do not flash past. Also writes PNGs under
 * {@code samples/out/show-demo/}.
 *
 * <pre>
 *   # GUI session (do NOT set -Djava.awt.headless=true)
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        -cp "target/samples-compile:target/classes:&lt;jars&gt;" \
 *        samples.ShowPlotDemo
 * </pre>
 *
 * <p>Close each window to advance to the next chart. Use
 * {@code chart.show(false)} if you want non-blocking multi-window display.
 */
public class ShowPlotDemo {
    public static void main(String[] args) throws Exception {
        boolean headless = java.awt.GraphicsEnvironment.isHeadless();
        System.out.println("headless=" + headless
            + "  (if true, show() saves a temp PNG instead of opening a window)");
        System.out.println("Each show() blocks until you close the window.\n");

        NP.Random.seed(42);
        java.nio.file.Path out = java.nio.file.Paths.get("samples/out/show-demo");
        java.nio.file.Files.createDirectories(out);

        // 1) Matplotlib line — blocking show
        NDArray x = NP.linspace(0, 10, 200);
        double[] xx = x.asDoubleArray();
        double[] yy = new double[xx.length];
        for (int i = 0; i < xx.length; i++) {
            yy[i] = Math.sin(xx[i]) + 0.1 * NP.Random.randn(1).getDouble(0);
        }
        LineChart line = Matplotlib.plot(x, new NDArray(yy, yy.length), "sin+noise")
            .setTitle("Demo 1/3 — Matplotlib.plot (close window to continue)")
            .setXAxisLabel("x").setYAxisLabel("y")
            .setShowGrid(true).setShowLegend(true)
            .setSize(800, 500);
        line.savefig(out.resolve("01_line.png").toString());
        System.out.println("saved " + out.resolve("01_line.png").toAbsolutePath());
        System.out.println("→ showing Demo 1 (blocking)…");
        line.show(); // block until close
        System.out.println("← Demo 1 closed");

        // 2) Seaborn hist — blocking show
        var hist = Seaborn.histplot(NP.Random.normal(0, 1, 1000), 30, true)
            .setTitle("Demo 2/3 — Seaborn.histplot (close window to continue)")
            .setSize(800, 500);
        hist.savefig(out.resolve("02_hist.png").toString());
        System.out.println("saved " + out.resolve("02_hist.png").toAbsolutePath());
        System.out.println("→ showing Demo 2 (blocking)…");
        hist.show();
        System.out.println("← Demo 2 closed");

        // 3) Non-blocking multi-window then wait on last
        var s1 = Matplotlib.scatter(NP.Random.randn(200), NP.Random.randn(200))
            .setTitle("Demo 3a — non-blocking scatter").setAlpha(0.6);
        var s2 = Matplotlib.bar(new String[]{"A", "B", "C"}, new double[]{3, 7, 4})
            .setTitle("Demo 3b — non-blocking bar");
        s1.savefig(out.resolve("03a_scatter.png").toString());
        s2.savefig(out.resolve("03b_bar.png").toString());
        System.out.println("→ opening Demo 3a + 3b non-blocking, then blocking on 3b…");
        s1.show(false); // stays open
        s2.show(true);  // block on this one
        System.out.println("← Demo 3b closed; open non-modal windows: "
            + BaseChart.openWindowCount());
        Matplotlib.close();
        System.out.println("Done. PNGs under " + out.toAbsolutePath());
    }
}
