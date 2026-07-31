package org.bytedeco.pytorch.plot.swanlab;

import org.bytedeco.pytorch.Tensor;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Training-loop helper for SwanLab (mirrors {@code swanlab.log} usage).
 */
public final class SwanLabTrainingMonitor implements AutoCloseable {

    private final SwanLabClient client;
    private final String experiment;
    private long step;
    private final boolean closeClient;

    public SwanLabTrainingMonitor(SwanLabClient client)
            throws IOException, InterruptedException {
        this(client, Map.of("framework", "javacpp-pytorch"), false);
    }

    public SwanLabTrainingMonitor(SwanLabClient client, Map<String, ?> config, boolean closeClient)
            throws IOException, InterruptedException {
        this.client = client;
        this.experiment = client.experiment();
        this.closeClient = closeClient;
        this.step = 0;
        client.init(config);
    }

    public SwanLabClient client() { return client; }
    public long step() { return step; }
    public String uiUrl() { return client.uiUrl(); }

    public void logMetric(String key, double value) throws IOException, InterruptedException {
        client.log(Map.of(key, value), step++);
    }

    public void logLoss(double loss) throws IOException, InterruptedException {
        logMetric("loss", loss);
    }

    public void logAccuracy(double accuracy) throws IOException, InterruptedException {
        logMetric("accuracy", accuracy);
    }

    public void log(Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        client.log(metrics, step++);
    }

    public void logHeatmap(String name, double[][] matrix) throws IOException, InterruptedException {
        client.logHeatmap(name, matrix, step);
    }

    public void logHeatmap(String name, Tensor t) throws IOException, InterruptedException {
        client.logHeatmap(name, t, step);
    }

    public void logImage(String name, Tensor image) throws IOException, InterruptedException {
        client.logImage(name, image, step);
    }

    public void logHistogram(String name, double[] values) throws IOException, InterruptedException {
        client.logHistogram(name, values, 30, step);
    }

    public void logText(String name, String text) throws IOException, InterruptedException {
        client.logText(name, text, step);
    }

    public void setStep(long step) { this.step = step; }

    @Override
    public void close() {
        try {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("final_step", step);
            summary.put("experiment", experiment);
            client.logSummary(summary);
            client.finish();
        } catch (Exception ignored) {
        }
        if (closeClient) client.close();
    }
}
