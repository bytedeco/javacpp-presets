package org.bytedeco.pytorch.plot.visdom;

import org.bytedeco.pytorch.Tensor;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * High-level training helper that streams loss / accuracy / histograms / images
 * into a running Visdom dashboard.
 *
 * <pre>{@code
 * try (VisdomClient viz = VisdomClient.newBuilder().env("train").build();
 *      VisdomTrainingMonitor mon = new VisdomTrainingMonitor(viz, "exp1")) {
 *     mon.logLoss(step, loss);
 *     mon.logAccuracy(step, acc);
 *     mon.logHistogram("weights", weightTensor);
 *     mon.logImage("sample", imageTensor);
 * }
 * }</pre>
 */
public final class VisdomTrainingMonitor implements AutoCloseable {

    private final VisdomClient client;
    private final String runName;
    private final String lossWin;
    private final String accWin;
    private final String lrWin;
    private final boolean closeClient;
    private boolean closed;

    public VisdomTrainingMonitor(VisdomClient client, String runName) {
        this(client, runName, false);
    }

    public VisdomTrainingMonitor(VisdomClient client, String runName, boolean closeClient) {
        this.client = client;
        this.runName = runName == null ? "run" : runName;
        this.lossWin = this.runName + "-loss";
        this.accWin = this.runName + "-accuracy";
        this.lrWin = this.runName + "-lr";
        this.closeClient = closeClient;
    }

    public VisdomClient client() { return client; }
    public String runName() { return runName; }

    public void logLoss(long step, double loss) {
        try {
            client.lineAppend(lossWin, step, loss, "loss",
                    VisdomClient.opts("title", runName + " / Loss",
                            "xlabel", "step", "ylabel", "loss", "showlegend", true));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logAccuracy(long step, double accuracy) {
        try {
            client.lineAppend(accWin, step, accuracy, "acc",
                    VisdomClient.opts("title", runName + " / Accuracy",
                            "xlabel", "step", "ylabel", "accuracy", "showlegend", true));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logLearningRate(long step, double lr) {
        try {
            client.lineAppend(lrWin, step, lr, "lr",
                    VisdomClient.opts("title", runName + " / LR",
                            "xlabel", "step", "ylabel", "lr"));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logMetric(String name, long step, double value) {
        try {
            client.lineAppend(runName + "-" + name, step, value, name,
                    VisdomClient.opts("title", runName + " / " + name,
                            "xlabel", "step", "ylabel", name));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logMetrics(long step, Map<String, ? extends Number> metrics) {
        if (metrics == null) return;
        for (Map.Entry<String, ? extends Number> e : metrics.entrySet()) {
            logMetric(e.getKey(), step, e.getValue().doubleValue());
        }
    }

    public void logHistogram(String name, double[] values) {
        try {
            client.histogram(values, runName + "-hist-" + name,
                    VisdomClient.opts("title", runName + " / hist " + name));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logHistogram(String name, Tensor t) {
        float[] f = VisdomClient.tensorToFloat(t);
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) d[i] = f[i];
        logHistogram(name, d);
    }

    public void logHeatmap(String name, double[][] matrix, Map<String, Object> opts) {
        try {
            Map<String, Object> o = opts == null ? new LinkedHashMap<>() : new LinkedHashMap<>(opts);
            o.putIfAbsent("title", runName + " / " + name);
            o.putIfAbsent("colormap", "Viridis");
            client.heatmap(matrix, runName + "-hm-" + name, o);
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logHeatmap(String name, Tensor t) {
        try {
            client.heatmap(t, runName + "-hm-" + name,
                    VisdomClient.opts("title", runName + " / " + name, "colormap", "Viridis"));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logImage(String name, Tensor image) {
        try {
            client.image(image, runName + "-img-" + name,
                    VisdomClient.opts("title", runName + " / " + name, "caption", name));
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logText(String name, String content) {
        try {
            client.text(content, runName + "-txt-" + name,
                    VisdomClient.opts("title", runName + " / " + name), false);
        } catch (IOException | InterruptedException e) {
            rethrow(e);
        }
    }

    public void logModelSummary(String summary) {
        logText("model", summary);
    }

    private static void rethrow(Exception e) {
        if (e instanceof InterruptedException) Thread.currentThread().interrupt();
        throw new RuntimeException("VisdomTrainingMonitor failed: " + e.getMessage(), e);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (closeClient) client.close();
    }
}
