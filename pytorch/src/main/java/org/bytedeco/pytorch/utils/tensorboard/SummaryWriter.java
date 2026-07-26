package org.bytedeco.pytorch.utils.tensorboard;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.utils.tensorboard.ProtoWire.*;

/**
 * TensorBoard {@code SummaryWriter} for JavaCPP / LibTorch.
 *
 * <p>API mirrors {@code torch.utils.tensorboard.SummaryWriter}: Tensor-first,
 * minimal surface. Low-level PNG/WAV/TFRecord details stay internal.
 *
 * <pre>{@code
 * try (SummaryWriter w = new SummaryWriter("runs/exp1")) {
 *     w.add_scalar("train/loss", loss.item_float(), step);
 *     w.add_histogram("fc1.weight", model.fc1.weight(), step);
 *     w.add_image("samples", images[0], step);          // CHW tensor
 *     w.add_images("batch", images, step);              // NCHW tensor
 *     w.add_audio("wav", waveform, step, 16000);
 *     w.add_text("note", "epoch done", step);
 *     w.add_pr_curve("pr", labels, preds, step);
 *     w.add_hparams(Map.of("lr", 1e-3), Map.of("hparam/acc", acc));
 * }
 * // tensorboard --logdir runs
 * }</pre>
 *
 * <p>Snake_case ({@code add_scalar}) and camelCase ({@code addScalar}) are both
 * provided; prefer snake_case for drop-in parity with Python tutorials.
 */
public final class SummaryWriter implements AutoCloseable {

    private final String logDir;
    private final OutputStream out;
    private final String path;
    private boolean closed;
    private final List<String> projectorEmbeddings = new ArrayList<>();

    // ------------------------------------------------------------------ ctors

    /** {@code SummaryWriter(log_dir)} — event file name matches PyTorch. */
    public SummaryWriter(String logDir) throws IOException {
        this(logDir, defaultEventFileName());
    }

    public SummaryWriter(String logDir, String fileName) throws IOException {
        this.logDir = Objects.requireNonNull(logDir, "logDir");
        Files.createDirectories(new File(logDir).toPath());
        this.path = logDir + File.separator + fileName;
        this.out = new FileOutputStream(this.path);
        writeEvent(buildFileVersionEvent("brain.Event:2"));
    }

    public String get_logdir() { return logDir; }
    public String logDir() { return logDir; }
    public String path() { return path; }

    static String defaultEventFileName() {
        String host = "localhost";
        try { host = java.net.InetAddress.getLocalHost().getHostName(); }
        catch (Exception ignored) { /* keep */ }
        long sec = System.currentTimeMillis() / 1000L;
        long pid;
        try { pid = ProcessHandle.current().pid(); }
        catch (Throwable t) { pid = 0L; }
        // PyTorch: events.out.tfevents.<time>.<host>.<pid>
        return "events.out.tfevents." + sec + "." + host + "." + pid;
    }

    // =========================================================================
    // add_scalar / add_scalars
    // =========================================================================

    /**
     * Log a scalar. {@code scalar_value} may be a Java {@link Number} or a
     * 0-dim / 1-element {@link Tensor} (matches Python).
     */
    public void add_scalar(String tag, Object scalarValue, long globalStep) throws IOException {
        writeSummary(Summaries.scalar(tag, toFloatScalar(scalarValue)), globalStep);
    }

    public void add_scalar(String tag, Object scalarValue) throws IOException {
        add_scalar(tag, scalarValue, 0L);
    }

    public void addScalar(String tag, Object scalarValue, long globalStep) throws IOException {
        add_scalar(tag, scalarValue, globalStep);
    }

    public void add_scalars(String mainTag, Map<String, ?> tagScalarDict, long globalStep) throws IOException {
        for (Map.Entry<String, ?> e : tagScalarDict.entrySet()) {
            add_scalar(mainTag + "/" + e.getKey(), e.getValue(), globalStep);
        }
    }

    public void add_scalars(String mainTag, Map<String, ?> tagScalarDict) throws IOException {
        add_scalars(mainTag, tagScalarDict, 0L);
    }

    public void addScalars(String mainTag, Map<String, ?> tagScalarDict, long globalStep) throws IOException {
        add_scalars(mainTag, tagScalarDict, globalStep);
    }

    // =========================================================================
    // add_histogram
    // =========================================================================

    /** Histogram from a Tensor (any shape; flattened) or {@code double[]}/{@code float[]}. */
    public void add_histogram(String tag, Object values, long globalStep) throws IOException {
        writeSummary(Summaries.histogram(tag, toDoubleArray(values)), globalStep);
    }

    public void add_histogram(String tag, Object values) throws IOException {
        add_histogram(tag, values, 0L);
    }

    public void addHistogram(String tag, Object values, long globalStep) throws IOException {
        add_histogram(tag, values, globalStep);
    }

    public void add_histogram_raw(String tag,
                                  double min, double max, double num,
                                  double sum, double sumSquares,
                                  double[] bucketLimits, double[] bucketCounts,
                                  long globalStep) throws IOException {
        writeSummary(Summaries.histogramRaw(tag, min, max, num, sum, sumSquares,
                bucketLimits, bucketCounts), globalStep);
    }

    public void addHistogramRaw(String tag,
                                double min, double max, double num,
                                double sum, double sumSquares,
                                double[] bucketLimits, double[] bucketCounts,
                                long globalStep) throws IOException {
        add_histogram_raw(tag, min, max, num, sum, sumSquares, bucketLimits, bucketCounts, globalStep);
    }

    // =========================================================================
    // add_image / add_images
    // =========================================================================

    /**
     * Log one image.
     * @param imgTensor rank-3 CHW (default) or HWC depending on {@code dataformats}
     */
    public void add_image(String tag, Tensor imgTensor, long globalStep, String dataformats) throws IOException {
        ImageBuffer img = tensorToHWC(imgTensor, dataformats == null ? "CHW" : dataformats);
        writeSummary(Summaries.imageFloatHWC(tag, img.hwc, img.h, img.w, img.c), globalStep);
    }

    public void add_image(String tag, Tensor imgTensor, long globalStep) throws IOException {
        add_image(tag, imgTensor, globalStep, "CHW");
    }

    public void add_image(String tag, Tensor imgTensor) throws IOException {
        add_image(tag, imgTensor, 0L, "CHW");
    }

    public void addImage(String tag, Tensor imgTensor, long globalStep) throws IOException {
        add_image(tag, imgTensor, globalStep, "CHW");
    }

    public void addImage(String tag, Tensor imgTensor, long globalStep, String dataformats) throws IOException {
        add_image(tag, imgTensor, globalStep, dataformats);
    }

    /**
     * Log a batch of images.
     * @param imgTensor rank-4 NCHW (default) or NHWC
     */
    public void add_images(String tag, Tensor imgTensor, long globalStep, String dataformats) throws IOException {
        List<ImageBuffer> imgs = tensorBatchToHWC(imgTensor, dataformats == null ? "NCHW" : dataformats);
        if (imgs.size() == 1) {
            ImageBuffer im = imgs.get(0);
            writeSummary(Summaries.imageFloatHWC(tag + "/image", im.hwc, im.h, im.w, im.c), globalStep);
        } else {
            for (int i = 0; i < imgs.size(); i++) {
                ImageBuffer im = imgs.get(i);
                writeSummary(Summaries.imageFloatHWC(tag + "/image/" + i, im.hwc, im.h, im.w, im.c), globalStep);
            }
        }
    }

    public void add_images(String tag, Tensor imgTensor, long globalStep) throws IOException {
        add_images(tag, imgTensor, globalStep, "NCHW");
    }

    public void add_images(String tag, Tensor imgTensor) throws IOException {
        add_images(tag, imgTensor, 0L, "NCHW");
    }

    public void addImages(String tag, Tensor imgTensor, long globalStep) throws IOException {
        add_images(tag, imgTensor, globalStep, "NCHW");
    }

    public void addImages(String tag, Tensor imgTensor, long globalStep, String dataformats) throws IOException {
        add_images(tag, imgTensor, globalStep, dataformats);
    }

    /**
     * {@code add_figure} equivalent: pass a chart already rendered to a CHW/HWC image tensor
     * (Java has no matplotlib). Prefer {@link #add_image}.
     */
    public void add_figure(String tag, Tensor figureImage, long globalStep) throws IOException {
        add_image(tag, figureImage, globalStep, figureImage.dim() == 3 && figureImage.size(0) <= 4 ? "CHW" : "HWC");
    }

    public void addFigure(String tag, Tensor figureImage, long globalStep) throws IOException {
        add_figure(tag, figureImage, globalStep);
    }

    // =========================================================================
    // add_video
    // =========================================================================

    /**
     * Log video frames as sequential images {@code tag/frame/i}
     * (full GIF encode needs an external encoder; Tensor path matches training use).
     * @param vidTensor {@code TCHW} or {@code NTCHW} (N=1)
     */
    public void add_video(String tag, Tensor vidTensor, long globalStep, int fps) throws IOException {
        Tensor v = vidTensor.contiguous().cpu();
        if (v.dim() == 5 && v.size(0) == 1) v = v.squeeze(0);
        if (v.dim() != 4) {
            throw new IllegalArgumentException("add_video expects TCHW or NTCHW tensor, got dim=" + v.dim());
        }
        int T = (int) v.size(0);
        // fps reserved for future GIF path; frames are still stepped for TB image plugin
        for (int t = 0; t < T; t++) {
            add_image(tag + "/frame/" + t, v.select(0, t), globalStep, "CHW");
        }
    }

    public void add_video(String tag, Tensor vidTensor, long globalStep) throws IOException {
        add_video(tag, vidTensor, globalStep, 4);
    }

    public void addVideo(String tag, Tensor vidTensor, long globalStep) throws IOException {
        add_video(tag, vidTensor, globalStep, 4);
    }

    // =========================================================================
    // add_audio
    // =========================================================================

    /**
     * @param sndTensor 1-D waveform (or squeezed), values in roughly [-1, 1]
     * @param sampleRate Hz, default 44100 like PyTorch
     */
    public void add_audio(String tag, Tensor sndTensor, long globalStep, int sampleRate) throws IOException {
        float[] samples = toFloatArray(sndTensor.flatten());
        writeSummary(Summaries.audio(tag, samples, sampleRate), globalStep);
    }

    public void add_audio(String tag, Tensor sndTensor, long globalStep) throws IOException {
        add_audio(tag, sndTensor, globalStep, 44100);
    }

    public void add_audio(String tag, Tensor sndTensor) throws IOException {
        add_audio(tag, sndTensor, 0L, 44100);
    }

    public void addAudio(String tag, Tensor sndTensor, long globalStep, int sampleRate) throws IOException {
        add_audio(tag, sndTensor, globalStep, sampleRate);
    }

    // =========================================================================
    // add_text / add_tensor
    // =========================================================================

    public void add_text(String tag, String textString, long globalStep) throws IOException {
        writeSummary(Summaries.text(tag, textString), globalStep);
    }

    public void add_text(String tag, String textString) throws IOException {
        add_text(tag, textString, 0L);
    }

    public void addText(String tag, String textString, long globalStep) throws IOException {
        add_text(tag, textString, globalStep);
    }

    public void add_tensor(String tag, Tensor tensor, long globalStep) throws IOException {
        Tensor c = tensor.contiguous().cpu().to(torch.kFloat());
        writeSummary(Summaries.tensorFloat(tag, shapeOf(c), toFloatArray(c)), globalStep);
    }

    public void add_tensor(String tag, Tensor tensor) throws IOException {
        add_tensor(tag, tensor, 0L);
    }

    public void addTensor(String tag, Tensor tensor, long globalStep) throws IOException {
        add_tensor(tag, tensor, globalStep);
    }

    // =========================================================================
    // add_pr_curve
    // =========================================================================

    /** {@code labels} and {@code predictions} are 1-D tensors (or float arrays). */
    public void add_pr_curve(String tag, Object labels, Object predictions,
                             long globalStep, int numThresholds) throws IOException {
        float[] y = toFloatArray(labels);
        float[] p = toFloatArray(predictions);
        writeSummary(Summaries.prCurve(tag, y, p, numThresholds), globalStep);
    }

    public void add_pr_curve(String tag, Object labels, Object predictions, long globalStep) throws IOException {
        add_pr_curve(tag, labels, predictions, globalStep, 127);
    }

    public void add_pr_curve(String tag, Object labels, Object predictions) throws IOException {
        add_pr_curve(tag, labels, predictions, 0L, 127);
    }

    public void addPrCurve(String tag, Object labels, Object predictions, long globalStep) throws IOException {
        add_pr_curve(tag, labels, predictions, globalStep, 127);
    }

    public void add_pr_curve_raw(String tag,
                                 float[] tp, float[] fp, float[] tn, float[] fn,
                                 float[] precision, float[] recall,
                                 long globalStep, int numThresholds) throws IOException {
        int T = numThresholds;
        float[] stacked = new float[6 * T];
        System.arraycopy(tp, 0, stacked, 0 * T, T);
        System.arraycopy(fp, 0, stacked, 1 * T, T);
        System.arraycopy(tn, 0, stacked, 2 * T, T);
        System.arraycopy(fn, 0, stacked, 3 * T, T);
        System.arraycopy(precision, 0, stacked, 4 * T, T);
        System.arraycopy(recall, 0, stacked, 5 * T, T);
        writeSummary(Summaries.prCurveRaw(tag, stacked, T), globalStep);
    }

    // =========================================================================
    // add_hparams
    // =========================================================================

    /**
     * Same contract as PyTorch: writes a child run under {@code log_dir/<run_name>/}
     * with experiment / session_start / session_end + metric scalars.
     */
    public void add_hparams(Map<String, ?> hparamDict,
                            Map<String, ? extends Number> metricDict,
                            String runName,
                            Long globalStep) throws IOException {
        if (runName == null || runName.isEmpty()) runName = Long.toString(System.currentTimeMillis());
        long step = globalStep == null ? 0L : globalStep;
        Summaries.HparamsBundle bundle = Summaries.hparams(hparamDict, metricDict);
        String childDir = logDir + File.separator + runName;
        try (SummaryWriter w = new SummaryWriter(childDir)) {
            w.writeSummary(bundle.experiment(), step);
            w.writeSummary(bundle.sessionStart(), step);
            w.writeSummary(bundle.sessionEnd(), step);
            for (Map.Entry<String, ? extends Number> e : metricDict.entrySet()) {
                w.add_scalar(e.getKey(), e.getValue(), step);
            }
        }
    }

    public void add_hparams(Map<String, ?> hparamDict, Map<String, ? extends Number> metricDict) throws IOException {
        add_hparams(hparamDict, metricDict, null, null);
    }

    public void addHparams(Map<String, ?> hparamDict, Map<String, ? extends Number> metricDict) throws IOException {
        add_hparams(hparamDict, metricDict, null, null);
    }

    // =========================================================================
    // add_mesh
    // =========================================================================

    /**
     * @param vertices Tensor {@code [B,N,3]} or {@code [N,3]} (auto-batched)
     * @param colors   optional {@code [B,N,3]} / {@code [N,3]} in 0..255
     * @param faces    optional {@code [B,M,3]} / {@code [M,3]}
     */
    public void add_mesh(String tag, Tensor vertices, Tensor colors, Tensor faces,
                         Map<String, ?> configDict, long globalStep) throws IOException {
        float[] v = flat3(vertices);
        long[] vs = shape3(vertices);
        float[] c = colors == null ? null : flat3(colors.to(torch.kFloat()));
        long[] cs = colors == null ? null : shape3(colors);
        float[] f = faces == null ? null : flat3(faces.to(torch.kFloat()));
        long[] fs = faces == null ? null : shape3(faces);
        String json = configDict == null || configDict.isEmpty() ? "{}" : toJsonObject(configDict);
        writeSummary(Summaries.mesh(tag, v, vs, f, fs, c, cs, json), globalStep);
    }

    public void add_mesh(String tag, Tensor vertices, Tensor colors, Tensor faces, long globalStep) throws IOException {
        add_mesh(tag, vertices, colors, faces, null, globalStep);
    }

    public void add_mesh(String tag, Tensor vertices, long globalStep) throws IOException {
        add_mesh(tag, vertices, null, null, null, globalStep);
    }

    public void addMesh(String tag, Tensor vertices, Tensor colors, Tensor faces, long globalStep) throws IOException {
        add_mesh(tag, vertices, colors, faces, null, globalStep);
    }

    // =========================================================================
    // add_heatmap  (2D matrix → colorized image; common TB training viz)
    // =========================================================================

    /**
     * Log a 2-D heatmap as an RGB image (TensorBoard has no separate heatmap
     * plugin; this matches the usual PyTorch practice of {@code add_image} on a
     * colormapped matrix — confusion matrix, attention, activation maps, …).
     *
     * @param values  rank-2 {@code [H,W]} (or rank-3 with C=1) tensor
     * @param cmap    {@code "viridis"} (default), {@code "jet"}, {@code "gray"}, {@code "hot"}
     */
    public void add_heatmap(String tag, Tensor values, long globalStep, String cmap) throws IOException {
        Tensor img = heatmapToRgb(values, cmap == null ? "viridis" : cmap);
        add_image(tag, img, globalStep, "CHW");
    }

    public void add_heatmap(String tag, Tensor values, long globalStep) throws IOException {
        add_heatmap(tag, values, globalStep, "viridis");
    }

    public void add_heatmap(String tag, Tensor values) throws IOException {
        add_heatmap(tag, values, 0L, "viridis");
    }

    public void addHeatmap(String tag, Tensor values, long globalStep) throws IOException {
        add_heatmap(tag, values, globalStep, "viridis");
    }

    public void addHeatmap(String tag, Tensor values, long globalStep, String cmap) throws IOException {
        add_heatmap(tag, values, globalStep, cmap);
    }

    // =========================================================================
    // add_embedding
    // =========================================================================

    /**
     * Projector embedding.
     * @param mat          {@code [N,D]} feature matrix
     * @param metadata     optional list of N labels (String) or N rows (List)
     * @param labelImg     optional {@code [N,C,H,W]} sprite images
     * @param metadataHeader optional header when metadata is multi-column
     */
    public void add_embedding(Tensor mat,
                              List<?> metadata,
                              Tensor labelImg,
                              long globalStep,
                              String tag,
                              List<String> metadataHeader) throws IOException {
        if (tag == null || tag.isEmpty()) tag = "default";
        float[][] rows = tensorToRows(mat);
        byte[][] sprites = null;
        int ih = 0, iw = 0, ic = 0;
        if (labelImg != null && labelImg.defined()) {
            Tensor img = labelImg.contiguous().cpu().to(torch.kFloat());
            if (img.dim() != 4) throw new IllegalArgumentException("label_img must be NCHW");
            int n = (int) img.size(0);
            ic = (int) img.size(1);
            ih = (int) img.size(2);
            iw = (int) img.size(3);
            sprites = new byte[n][];
            for (int i = 0; i < n; i++) {
                ImageBuffer hb = tensorToHWC(img.select(0, i), "CHW");
                sprites[i] = floatHwcToU8(hb.hwc);
            }
        }
        writeEmbeddingFiles(rows, metadata, metadataHeader, sprites, ih, iw, ic <= 0 ? 3 : ic, tag, globalStep);
    }


    public void add_embedding(Tensor mat, List<?> metadata, Tensor labelImg, long globalStep, String tag) throws IOException {
        add_embedding(mat, metadata, labelImg, globalStep, tag, null);
    }

    public void addEmbedding(Tensor mat, List<?> metadata, Tensor labelImg, long globalStep, String tag) throws IOException {
        add_embedding(mat, metadata, labelImg, globalStep, tag, null);
    }

    public void add_embedding(Tensor mat, List<?> metadata, long globalStep, String tag) throws IOException {
        add_embedding(mat, metadata, null, globalStep, tag, null);
    }

    public void add_embedding(Tensor mat, long globalStep, String tag) throws IOException {
        add_embedding(mat, null, null, globalStep, tag, null);
    }

    public void add_embedding(Tensor mat, long globalStep) throws IOException {
        add_embedding(mat, null, null, globalStep, "default", null);
    }

    public void addEmbedding(Tensor mat, List<?> metadata, long globalStep, String tag) throws IOException {
        add_embedding(mat, metadata, null, globalStep, tag, null);
    }

    // =========================================================================
    // add_graph / add_onnx_graph  (byte-level; full JIT trace is separate)
    // =========================================================================

    /** Write a raw GraphDef protobuf payload as {@code Event.graph_def}. */
    public void add_graph(byte[] graphDefBytes) throws IOException {
        ByteArrayOutputStream event = buf();
        double64(event, 1, System.currentTimeMillis() / 1000.0);
        int64(event, 2, 0);
        bytes(event, 4, graphDefBytes);
        writeEvent(event.toByteArray());
    }

    public void addGraph(byte[] graphDefBytes) throws IOException {
        add_graph(graphDefBytes);
    }

    /** Store ONNX model bytes under graph_def (convert to GraphDef for full UI). */
    public void add_onnx_graph(byte[] onnxModelBytes) throws IOException {
        add_graph(onnxModelBytes == null ? new byte[0] : onnxModelBytes);
    }

    public void add_onnx_graph(File onnxFile) throws IOException {
        add_onnx_graph(Files.readAllBytes(onnxFile.toPath()));
    }

    public void addOnnxGraph(byte[] onnxModelBytes) throws IOException {
        add_onnx_graph(onnxModelBytes);
    }

    // =========================================================================
    // add_custom_scalars
    // =========================================================================

    /**
     * @param layout category → (chartName → Object[]{ "Multiline"|"Margin", List&lt;String&gt; tags })
     */
    public void add_custom_scalars(Map<String, Map<String, Object[]>> layout) throws IOException {
        writeSummary(Summaries.customScalars(layout), 0);
    }

    public void addCustomScalars(Map<String, Map<String, Object[]>> layout) throws IOException {
        add_custom_scalars(layout);
    }

    public void add_custom_scalars_multilinechart(List<String> tags, String category, String title) throws IOException {
        Map<String, Map<String, Object[]>> layout = new LinkedHashMap<>();
        Map<String, Object[]> charts = new LinkedHashMap<>();
        charts.put(title == null ? "untitled" : title, new Object[]{"Multiline", tags});
        layout.put(category == null ? "default" : category, charts);
        add_custom_scalars(layout);
    }

    public void add_custom_scalars_marginchart(List<String> tags, String category, String title) throws IOException {
        if (tags == null || tags.size() != 3) {
            throw new IllegalArgumentException("margin chart needs exactly 3 tags: value, lower, upper");
        }
        Map<String, Map<String, Object[]>> layout = new LinkedHashMap<>();
        Map<String, Object[]> charts = new LinkedHashMap<>();
        charts.put(title == null ? "untitled" : title, new Object[]{"Margin", tags});
        layout.put(category == null ? "default" : category, charts);
        add_custom_scalars(layout);
    }

    // =========================================================================
    // lifecycle
    // =========================================================================

    public void flush() throws IOException { out.flush(); }

    @Override
    public void close() throws IOException {
        if (closed) return;
        closed = true;
        out.flush();
        out.close();
    }

    // =========================================================================
    // internals (not part of the PyTorch-facing surface)
    // =========================================================================

    void writeSummary(byte[] summaryProto, long step) throws IOException {
        ByteArrayOutputStream event = buf();
        double64(event, 1, System.currentTimeMillis() / 1000.0);
        int64(event, 2, step);
        message(event, 5, summaryProto);
        writeEvent(event.toByteArray());
    }

    private void writeEvent(byte[] eventProto) throws IOException {
        byte[] len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(eventProto.length).array();
        out.write(len);
        writeIntLE(Crc32C.maskedCrc32c(len));
        out.write(eventProto);
        writeIntLE(Crc32C.maskedCrc32c(eventProto));
    }

    private void writeIntLE(int v) throws IOException {
        out.write(v & 0xff);
        out.write((v >>> 8) & 0xff);
        out.write((v >>> 16) & 0xff);
        out.write((v >>> 24) & 0xff);
    }

    private static byte[] buildFileVersionEvent(String version) {
        ByteArrayOutputStream bos = buf();
        double64(bos, 1, System.currentTimeMillis() / 1000.0);
        int64(bos, 2, 0);
        string(bos, 3, version);
        ByteArrayOutputStream sm = buf();
        string(sm, 1, "org.bytedeco.pytorch.utils.tensorboard.SummaryWriter");
        message(bos, 10, sm.toByteArray());
        return bos.toByteArray();
    }

    private void writeEmbeddingFiles(float[][] mat, List<?> metadata, List<String> metadataHeader,
                                     byte[][] labelImagesHWC, int imageH, int imageW, int imageC,
                                     String tag, long globalStep) throws IOException {
        String subdir = String.format("%05d/%s", globalStep, encodeTag(tag));
        File savePath = new File(logDir, subdir);
        Files.createDirectories(savePath.toPath());

        try (OutputStream os = new FileOutputStream(new File(savePath, "tensors.tsv"))) {
            for (float[] row : mat) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    if (i > 0) sb.append('\t');
                    sb.append(row[i]);
                }
                sb.append('\n');
                os.write(sb.toString().getBytes(StandardCharsets.UTF_8));
            }
        }
        if (metadata != null) {
            if (metadata.size() != mat.length) {
                throw new IllegalArgumentException("#labels should equal #data points");
            }
            try (OutputStream os = new FileOutputStream(new File(savePath, "metadata.tsv"))) {
                if (metadataHeader != null && !metadataHeader.isEmpty()) {
                    os.write(String.join("\t", metadataHeader).concat("\n").getBytes(StandardCharsets.UTF_8));
                }
                for (Object row : metadata) {
                    String line;
                    if (row instanceof List<?> list) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = 0; i < list.size(); i++) {
                            if (i > 0) sb.append('\t');
                            sb.append(list.get(i));
                        }
                        line = sb.toString();
                    } else {
                        line = String.valueOf(row);
                    }
                    os.write((line + "\n").getBytes(StandardCharsets.UTF_8));
                }
            }
        }
        int spriteW = 0, spriteH = 0;
        if (labelImagesHWC != null) {
            PngEncoder.Sprite sprite = PngEncoder.makeSpriteHWC(labelImagesHWC, imageH, imageW, imageC);
            Files.write(new File(savePath, "sprite.png").toPath(), sprite.png());
            spriteW = sprite.singleWidth();
            spriteH = sprite.singleHeight();
        }
        String tensorName = tag + ":" + String.format("%05d", globalStep);
        StringBuilder emb = new StringBuilder();
        emb.append("embeddings {\n");
        emb.append("  tensor_name: \"").append(escapeProto(tensorName)).append("\"\n");
        emb.append("  tensor_path: \"").append(escapeProto(subdir + "/tensors.tsv")).append("\"\n");
        if (metadata != null) {
            emb.append("  metadata_path: \"").append(escapeProto(subdir + "/metadata.tsv")).append("\"\n");
        }
        if (labelImagesHWC != null) {
            emb.append("  sprite {\n");
            emb.append("    image_path: \"").append(escapeProto(subdir + "/sprite.png")).append("\"\n");
            emb.append("    single_image_dim: ").append(spriteW).append("\n");
            emb.append("    single_image_dim: ").append(spriteH).append("\n");
            emb.append("  }\n");
        }
        emb.append("}\n");
        projectorEmbeddings.add(emb.toString());
        StringBuilder cfg = new StringBuilder();
        for (String e : projectorEmbeddings) cfg.append(e);
        Files.writeString(new File(logDir, "projector_config.pbtxt").toPath(), cfg.toString(), StandardCharsets.UTF_8);
    }

    // ---- conversions --------------------------------------------------------

    private static float toFloatScalar(Object v) {
        if (v == null) return 0f;
        if (v instanceof Number n) return n.floatValue();
        if (v instanceof Tensor t) {
            Tensor c = t.contiguous().cpu().flatten();
            if (c.numel() != 1) throw new IllegalArgumentException("scalar tensor must have 1 element");
            return c.to(torch.kFloat()).item_float(); // item as float via data
        }
        throw new IllegalArgumentException("scalar must be Number or Tensor, got " + v.getClass());
    }

    private static double[] toDoubleArray(Object values) {
        if (values == null) return new double[0];
        if (values instanceof double[] d) return d;
        if (values instanceof float[] f) {
            double[] out = new double[f.length];
            for (int i = 0; i < f.length; i++) out[i] = f[i];
            return out;
        }
        if (values instanceof Tensor t) {
            if (!t.defined()) return new double[0];
            Tensor c = t.contiguous().cpu().to(torch.kDouble()).flatten();
            long n = c.numel();
            double[] data = new double[(int) Math.min(n, Integer.MAX_VALUE)];
            DoublePointer p = c.data_ptr_double();
            for (int i = 0; i < data.length; i++) data[i] = p.get(i);
            return data;
        }
        if (values instanceof Iterable<?> it) {
            ArrayList<Double> tmp = new ArrayList<>();
            for (Object o : it) tmp.add(((Number) o).doubleValue());
            double[] out = new double[tmp.size()];
            for (int i = 0; i < out.length; i++) out[i] = tmp.get(i);
            return out;
        }
        throw new IllegalArgumentException("histogram values must be Tensor, double[], float[], or Iterable");
    }

    private static float[] toFloatArray(Object values) {
        if (values == null) return new float[0];
        if (values instanceof float[] f) return f;
        if (values instanceof double[] d) {
            float[] out = new float[d.length];
            for (int i = 0; i < d.length; i++) out[i] = (float) d[i];
            return out;
        }
        if (values instanceof Tensor t) return toFloatArray(t);
        if (values instanceof Iterable<?> it) {
            ArrayList<Float> tmp = new ArrayList<>();
            for (Object o : it) tmp.add(((Number) o).floatValue());
            float[] out = new float[tmp.size()];
            for (int i = 0; i < out.length; i++) out[i] = tmp.get(i);
            return out;
        }
        throw new IllegalArgumentException("expected Tensor / float[] / double[] / Iterable");
    }

    private static float[] toFloatArray(Tensor tensor) {
        if (tensor == null || !tensor.defined()) return new float[0];
        Tensor c = tensor.contiguous().cpu().to(torch.kFloat()).flatten();
        long n = c.numel();
        float[] data = new float[(int) Math.min(n, Integer.MAX_VALUE)];
        FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < data.length; i++) data[i] = p.get(i);
        return data;
    }

    private static long[] shapeOf(Tensor t) {
        long[] sz = new long[(int) t.dim()];
        for (int i = 0; i < sz.length; i++) sz[i] = t.size(i);
        return sz;
    }

    private static float[] flat3(Tensor t) {
        Tensor c = t.contiguous().cpu().to(torch.kFloat());
        if (c.dim() == 2) c = c.unsqueeze(0); // N,3 → 1,N,3
        return toFloatArray(c);
    }

    private static long[] shape3(Tensor t) {
        Tensor c = t;
        if (c.dim() == 2) return new long[]{1, c.size(0), c.size(1)};
        return shapeOf(c);
    }

    private static float[][] tensorToRows(Tensor mat) {
        Tensor c = mat.contiguous().cpu().to(torch.kFloat());
        if (c.dim() != 2) throw new IllegalArgumentException("embedding mat must be 2D (N,D)");
        int n = (int) c.size(0);
        int d = (int) c.size(1);
        float[] flat = toFloatArray(c);
        float[][] rows = new float[n][d];
        for (int i = 0; i < n; i++) System.arraycopy(flat, i * d, rows[i], 0, d);
        return rows;
    }

    private static byte[] floatHwcToU8(float[] hwc) {
        float max = 0f;
        for (float v : hwc) {
            float a = Math.abs(v);
            if (a > max) max = a;
        }
        float scale = max <= 1.0001f ? 255f : 1f;
        byte[] u8 = new byte[hwc.length];
        for (int i = 0; i < hwc.length; i++) {
            float v = hwc[i] * scale;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            u8[i] = (byte) (int) (v + 0.5f);
        }
        return u8;
    }

    /**
     * Convert a 2-D (or 1-channel 3-D) tensor into an RGB CHW image via a simple
     * colormap. Values are min-max normalized to [0,1] first.
     */
    private static Tensor heatmapToRgb(Tensor values, String cmap) {
        Tensor t = values.contiguous().cpu().to(torch.kFloat());
        if (t.dim() == 3 && t.size(0) == 1) t = t.squeeze(0);
        if (t.dim() == 3 && t.size(2) == 1) t = t.squeeze(2);
        if (t.dim() != 2) {
            throw new IllegalArgumentException("add_heatmap expects rank-2 [H,W] tensor, got dim=" + t.dim());
        }
        int h = (int) t.size(0);
        int w = (int) t.size(1);
        float[] raw = toFloatArray(t);
        float lo = raw[0], hi = raw[0];
        for (float v : raw) {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        float span = hi - lo;
        if (!(span > 0f)) span = 1f;

        float[] rgb = new float[3 * h * w];
        String c = cmap == null ? "viridis" : cmap.toLowerCase();
        for (int i = 0; i < raw.length; i++) {
            float x = (raw[i] - lo) / span;
            if (x < 0) x = 0;
            if (x > 1) x = 1;
            float[] col = colormap(c, x);
            rgb[0 * h * w + i] = col[0];
            rgb[1 * h * w + i] = col[1];
            rgb[2 * h * w + i] = col[2];
        }
        // build CHW tensor without depending on torch.tensor(float[]) shape helpers here
        Tensor flat = null;
        // use existing public path: write via imageFloatHWC needs float HWC; we have CHW planar
        // Convert planar RGB to a Tensor via repeated from scalar stack is heavy; use Data pointer path:
        // simplest portable approach: create empty then copy — but empty+index put is awkward in JavaCPP.
        // Reuse torch.tensor(float[]) + reshape which samples already use.
        return org.bytedeco.pytorch.global.torch.tensor(rgb).reshape(3, h, w);
    }

    /** Piecewise colormaps approximating matplotlib viridis/jet/hot/gray. */
    private static float[] colormap(String name, float x) {
        return switch (name) {
            case "gray", "grey" -> new float[]{x, x, x};
            case "hot" -> new float[]{
                    Math.min(1f, x * 3f),
                    Math.min(1f, Math.max(0f, x * 3f - 1f)),
                    Math.min(1f, Math.max(0f, x * 3f - 2f))
            };
            case "jet" -> jet(x);
            default -> viridis(x); // viridis
        };
    }

    private static float[] jet(float x) {
        float r = clamp01(1.5f - Math.abs(4f * x - 3f));
        float g = clamp01(1.5f - Math.abs(4f * x - 2f));
        float b = clamp01(1.5f - Math.abs(4f * x - 1f));
        return new float[]{r, g, b};
    }

    /** Compact 5-stop viridis approximation. */
    private static float[] viridis(float x) {
        // stops at 0, 0.25, 0.5, 0.75, 1
        float[][] stops = {
                {0.267f, 0.005f, 0.329f},
                {0.230f, 0.322f, 0.546f},
                {0.128f, 0.567f, 0.551f},
                {0.369f, 0.789f, 0.383f},
                {0.993f, 0.906f, 0.144f}
        };
        float pos = x * (stops.length - 1);
        int i = (int) Math.floor(pos);
        if (i >= stops.length - 1) return stops[stops.length - 1].clone();
        if (i < 0) return stops[0].clone();
        float t = pos - i;
        float[] a = stops[i], b = stops[i + 1];
        return new float[]{
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t
        };
    }

    private static float clamp01(float v) {
        if (v < 0) return 0;
        if (v > 1) return 1;
        return v;
    }

    private static final class ImageBuffer {
        final float[] hwc;
        final int h, w, c;
        ImageBuffer(float[] hwc, int h, int w, int c) { this.hwc = hwc; this.h = h; this.w = w; this.c = c; }
    }

    private static ImageBuffer tensorToHWC(Tensor tensor, String dataformats) {
        Tensor c = tensor.contiguous().cpu().to(torch.kFloat());
        long[] sz = shapeOf(c);
        String fmt = dataformats == null ? "CHW" : dataformats.toUpperCase();
        if (sz.length == 4 && sz[0] == 1 && fmt.startsWith("N")) {
            c = c.squeeze(0);
            sz = shapeOf(c);
            fmt = fmt.substring(1);
        }
        if (sz.length != 3) {
            throw new IllegalArgumentException("image tensor rank must be 3 (or 4 with N=1), got " + sz.length);
        }
        int a = (int) sz[0], b = (int) sz[1], cc = (int) sz[2];
        float[] flat = toFloatArray(c);
        int h, w, ch;
        float[] hwc;
        switch (fmt) {
            case "HWC" -> { h = a; w = b; ch = cc; hwc = flat; }
            case "CHW" -> {
                ch = a; h = b; w = cc;
                hwc = new float[h * w * ch];
                for (int ci = 0; ci < ch; ci++)
                    for (int yi = 0; yi < h; yi++)
                        for (int xi = 0; xi < w; xi++)
                            hwc[(yi * w + xi) * ch + ci] = flat[ci * h * w + yi * w + xi];
            }
            default -> throw new IllegalArgumentException("unsupported dataformats: " + dataformats + " (use CHW or HWC)");
        }
        return new ImageBuffer(hwc, h, w, ch);
    }

    private static List<ImageBuffer> tensorBatchToHWC(Tensor tensor, String dataformats) {
        Tensor c = tensor.contiguous().cpu().to(torch.kFloat());
        long[] sz = shapeOf(c);
        String fmt = dataformats == null ? "NCHW" : dataformats.toUpperCase();
        List<ImageBuffer> out = new ArrayList<>();
        if (sz.length == 3) {
            out.add(tensorToHWC(c, fmt.startsWith("N") ? fmt.substring(1) : fmt));
            return out;
        }
        if (sz.length != 4) throw new IllegalArgumentException("batch image rank must be 4 (NCHW/NHWC)");
        int n = (int) sz[0];
        String per = fmt.startsWith("N") ? fmt.substring(1) : "CHW";
        for (int i = 0; i < n; i++) out.add(tensorToHWC(c.select(0, i), per));
        return out;
    }

    private static String encodeTag(String tag) {
        return tag.replaceAll("[^a-zA-Z0-9._-]+", "_");
    }

    private static String escapeProto(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /** Tiny JSON object encoder for mesh config_dict (string/number/bool/nested-map only). */
    @SuppressWarnings("unchecked")
    private static String toJsonObject(Map<String, ?> map) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, ?> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(e.getKey().replace("\"", "\\\"")).append("\":");
            Object v = e.getValue();
            if (v == null) sb.append("null");
            else if (v instanceof Number n) sb.append(n);
            else if (v instanceof Boolean b) sb.append(b);
            else if (v instanceof Map<?, ?> m) sb.append(toJsonObject((Map<String, ?>) m));
            else if (v instanceof List<?> list) {
                sb.append('[');
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) sb.append(',');
                    Object x = list.get(i);
                    if (x instanceof Number || x instanceof Boolean) sb.append(x);
                    else sb.append('"').append(String.valueOf(x).replace("\"", "\\\"")).append('"');
                }
                sb.append(']');
            } else sb.append('"').append(String.valueOf(v).replace("\"", "\\\"")).append('"');
        }
        sb.append('}');
        return sb.toString();
    }
}
