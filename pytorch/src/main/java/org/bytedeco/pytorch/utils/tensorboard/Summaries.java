package org.bytedeco.pytorch.utils.tensorboard;
import org.bytedeco.pytorch.jit.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.bytedeco.pytorch.utils.tensorboard.ProtoWire.*;

/**
 * Build serialized {@code Summary} protobuf messages for every common
 * TensorBoard plugin, matching PyTorch {@code torch.utils.tensorboard.summary}.
 *
 * <p>Each method returns the bytes of a {@code tensorflow.Summary} message
 * (NOT an Event). {@link SummaryWriter} wraps these into Events.
 */
public final class Summaries {
    private Summaries() {}

    // DataClass enum (SummaryMetadata)
    public static final int DATA_CLASS_UNKNOWN = 0;
    public static final int DATA_CLASS_SCALAR = 1;
    public static final int DATA_CLASS_TENSOR = 2;
    public static final int DATA_CLASS_BLOB_SEQUENCE = 3;

    // ---- scalar -------------------------------------------------------------

    /** Classic simple_value scalar (universally accepted). */
    public static byte[] scalar(String tag, float value) {
        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        float32(val, 2, value); // simple_value
        message(val, 9, pluginMetadata("scalars", null, DATA_CLASS_SCALAR));
        return summaryOf(val.toByteArray());
    }

    /** New-style tensor scalar (DT_FLOAT rank-0). */
    public static byte[] scalarTensor(String tag, float value) {
        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 8, tensorProtoFloat(new long[0], new float[]{value}));
        message(val, 9, pluginMetadata("scalars", null, DATA_CLASS_SCALAR));
        return summaryOf(val.toByteArray());
    }

    // ---- histogram ----------------------------------------------------------

    public static byte[] histogram(String tag, double[] values) {
        return histogram(tag, values, 30);
    }

    public static byte[] histogram(String tag, double[] values, int maxBins) {
        if (values == null || values.length == 0) {
            values = new double[]{0.0};
        }
        double min = values[0], max = values[0], sum = 0, sumsq = 0;
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumsq += v * v;
        }
        int bins = Math.min(Math.max(maxBins, 1), Math.max(values.length, 1));
        double width = (max - min) / bins;
        if (!(width > 0)) {
            width = 1e-12;
            max = min + width * bins;
        }
        double[] limits = new double[bins];
        double[] counts = new double[bins];
        for (int i = 0; i < bins; i++) limits[i] = min + (i + 1) * width;
        limits[bins - 1] = max;
        for (double v : values) {
            int b = (int) Math.min(bins - 1, Math.max(0, (v - min) / width));
            counts[b]++;
        }
        // left empty bin like TF histogram.cc / pytorch make_histogram
        double[] limOut = new double[bins + 1];
        double[] cntOut = new double[bins + 1];
        limOut[0] = min;
        System.arraycopy(limits, 0, limOut, 1, bins);
        cntOut[0] = 0;
        System.arraycopy(counts, 0, cntOut, 1, bins);

        return histogramRaw(tag, min, max, values.length, sum, sumsq, limOut, cntOut);
    }

    public static byte[] histogramRaw(String tag,
                                      double min, double max, double num,
                                      double sum, double sumSquares,
                                      double[] bucketLimits, double[] bucketCounts) {
        ByteArrayOutputStream histo = buf();
        double64(histo, 1, min);
        double64(histo, 2, max);
        double64(histo, 3, num);
        double64(histo, 4, sum);
        double64(histo, 5, sumSquares);
        repeatedDouble(histo, 6, bucketLimits);
        repeatedDouble(histo, 7, bucketCounts);

        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 5, histo.toByteArray()); // histo
        message(val, 9, pluginMetadata("histograms"));
        return summaryOf(val.toByteArray());
    }

    // ---- image --------------------------------------------------------------

    /**
     * Image from already-encoded PNG/GIF/JPEG bytes.
     * {@code colorspace}: 1=gray, 2=gray+A, 3=RGB, 4=RGBA.
     */
    public static byte[] imageEncoded(String tag, byte[] encoded, int height, int width, int colorspace) {
        ByteArrayOutputStream img = buf();
        int32(img, 1, height);
        int32(img, 2, width);
        int32(img, 3, colorspace);
        bytes(img, 4, encoded); // encoded_image_string

        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 4, img.toByteArray()); // image
        message(val, 9, pluginMetadata("images", null, DATA_CLASS_BLOB_SEQUENCE));
        return summaryOf(val.toByteArray());
    }

    /** HWC uint8 image → PNG summary. */
    public static byte[] imageHWC(String tag, byte[] hwc, int height, int width, int channels) {
        byte[] png = PngEncoder.encodeHWC(hwc, height, width, channels);
        return imageEncoded(tag, png, height, width, channels);
    }

    /** HWC float image (auto scale) → PNG summary. */
    public static byte[] imageFloatHWC(String tag, float[] hwc, int height, int width, int channels) {
        byte[] png = PngEncoder.encodeFloatHWC(hwc, height, width, channels);
        return imageEncoded(tag, png, height, width, channels);
    }

    // ---- audio --------------------------------------------------------------

    /**
     * Mono PCM float samples in [-1,1] → WAV-encoded audio summary.
     */
    public static byte[] audio(String tag, float[] monoSamples, float sampleRate) {
        if (monoSamples == null) monoSamples = new float[0];
        // clip
        float[] clipped = monoSamples;
        float peak = 0;
        for (float s : monoSamples) {
            float a = Math.abs(s);
            if (a > peak) peak = a;
        }
        if (peak > 1f) {
            clipped = monoSamples.clone();
            for (int i = 0; i < clipped.length; i++) {
                float v = clipped[i];
                if (v > 1) v = 1;
                if (v < -1) v = -1;
                clipped[i] = v;
            }
        }
        byte[] wav = encodeWavPcm16(clipped, Math.round(sampleRate), 1);

        ByteArrayOutputStream audio = buf();
        float32(audio, 1, sampleRate);
        int64(audio, 2, 1); // num_channels
        int64(audio, 3, clipped.length); // length_frames
        bytes(audio, 4, wav);
        string(audio, 5, "audio/wav");

        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 6, audio.toByteArray()); // audio field 6
        message(val, 9, pluginMetadata("audio", null, DATA_CLASS_BLOB_SEQUENCE));
        return summaryOf(val.toByteArray());
    }

    /** Raw encoded audio (e.g. already-WAV bytes). */
    public static byte[] audioEncoded(String tag, byte[] encoded, float sampleRate,
                                      int numChannels, long lengthFrames, String contentType) {
        ByteArrayOutputStream audio = buf();
        float32(audio, 1, sampleRate);
        int64(audio, 2, numChannels);
        int64(audio, 3, lengthFrames);
        bytes(audio, 4, encoded);
        string(audio, 5, contentType == null ? "audio/wav" : contentType);

        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 6, audio.toByteArray());
        message(val, 9, pluginMetadata("audio", null, DATA_CLASS_BLOB_SEQUENCE));
        return summaryOf(val.toByteArray());
    }

    // ---- text ---------------------------------------------------------------

    public static byte[] text(String tag, String text) {
        // TextPluginData { version = 0 } serializes to empty bytes under proto3 defaults;
        // match pytorch: plugin_name only (content empty).
        String textTag = tag.endsWith("/text_summary") ? tag : tag + "/text_summary";
        ByteArrayOutputStream val = buf();
        string(val, 1, textTag);
        message(val, 8, tensorProtoString(new long[]{1}, text == null ? "" : text));
        message(val, 9, pluginMetadata("text"));
        return summaryOf(val.toByteArray());
    }

    // ---- tensor -------------------------------------------------------------

    public static byte[] tensorFloat(String tag, long[] shape, float[] data) {
        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 8, tensorProtoFloat(shape, data));
        message(val, 9, pluginMetadata("tensor", null, DATA_CLASS_TENSOR));
        return summaryOf(val.toByteArray());
    }

    // ---- pr_curve -----------------------------------------------------------

    /**
     * labels / predictions are parallel arrays; labels in {0,1}, predictions in [0,1].
     * Returns stacked [tp,fp,tn,fn,precision,recall] of shape [6, num_thresholds].
     */
    public static byte[] prCurve(String tag, float[] labels, float[] predictions, int numThresholds) {
        numThresholds = Math.min(Math.max(numThresholds, 2), 127);
        float[] data = computePrCurve(labels, predictions, numThresholds, null);
        return prCurveRaw(tag, data, numThresholds);
    }

    public static byte[] prCurveRaw(String tag, float[] stacked6xT, int numThresholds) {
        numThresholds = Math.min(Math.max(numThresholds, 2), 127);
        // PrCurvePluginData { version=1:0, num_thresholds=2:int32 }
        ByteArrayOutputStream content = buf();
        int32(content, 1, 0);
        int32(content, 2, numThresholds);

        ByteArrayOutputStream val = buf();
        string(val, 1, tag);
        message(val, 8, tensorProtoFloat(new long[]{6, numThresholds}, stacked6xT));
        message(val, 9, pluginMetadata("pr_curves", content.toByteArray()));
        return summaryOf(val.toByteArray());
    }

    public static float[] computePrCurve(float[] labels, float[] predictions,
                                         int numThresholds, float[] weights) {
        final double MIN = 1e-7;
        int n = labels.length;
        double[] tpB = new double[numThresholds];
        double[] fpB = new double[numThresholds];
        for (int i = 0; i < n; i++) {
            float p = predictions[i];
            if (p < 0) p = 0;
            if (p > 1) p = 1;
            int bucket = (int) Math.floor(p * (numThresholds - 1));
            if (bucket >= numThresholds) bucket = numThresholds - 1;
            double w = weights == null ? 1.0 : weights[i];
            double y = labels[i];
            tpB[bucket] += y * w;
            fpB[bucket] += (1.0 - y) * w;
        }
        // reverse cumsum
        double[] tp = new double[numThresholds];
        double[] fp = new double[numThresholds];
        double runTp = 0, runFp = 0;
        for (int i = numThresholds - 1; i >= 0; i--) {
            runTp += tpB[i];
            runFp += fpB[i];
            tp[i] = runTp;
            fp[i] = runFp;
        }
        double[] tn = new double[numThresholds];
        double[] fn = new double[numThresholds];
        float[] out = new float[6 * numThresholds];
        for (int i = 0; i < numThresholds; i++) {
            tn[i] = fp[0] - fp[i];
            fn[i] = tp[0] - tp[i];
            double prec = tp[i] / Math.max(MIN, tp[i] + fp[i]);
            double rec = tp[i] / Math.max(MIN, tp[i] + fn[i]);
            out[0 * numThresholds + i] = (float) tp[i];
            out[1 * numThresholds + i] = (float) fp[i];
            out[2 * numThresholds + i] = (float) tn[i];
            out[3 * numThresholds + i] = (float) fn[i];
            out[4 * numThresholds + i] = (float) prec;
            out[5 * numThresholds + i] = (float) rec;
        }
        return out;
    }

    // ---- hparams ------------------------------------------------------------

    /**
     * Build the three hparams summaries (experiment / session_start / session_end)
     * as separate Summary messages. Caller writes each and also logs metric scalars
     * into a child run directory (see {@link SummaryWriter#addHparams}).
     */
    public static HparamsBundle hparams(Map<String, ?> hparamDict, Map<String, ? extends Number> metricDict) {
        return hparams(hparamDict, metricDict, null);
    }

    public static HparamsBundle hparams(Map<String, ?> hparamDict,
                                        Map<String, ? extends Number> metricDict,
                                        Map<String, ? extends List<?>> domainDiscrete) {
        if (hparamDict == null || metricDict == null) {
            throw new IllegalArgumentException("hparamDict and metricDict required");
        }

        // ---- Experiment (api_pb2.Experiment inside HParamsPluginData.experiment) ----
        ByteArrayOutputStream experiment = buf();
        // hparam_infos (field 4), metric_infos (field 5)
        for (Map.Entry<String, ?> e : hparamDict.entrySet()) {
            if (e.getValue() == null) continue;
            ByteArrayOutputStream info = buf();
            string(info, 1, e.getKey()); // name
            int type = hparamType(e.getValue());
            enum_(info, 4, type); // type
            if (domainDiscrete != null && domainDiscrete.containsKey(e.getKey())) {
                List<?> dom = domainDiscrete.get(e.getKey());
                ByteArrayOutputStream list = buf();
                for (Object d : dom) message(list, 1, structValue(d));
                message(info, 5, list.toByteArray()); // domain_discrete = ListValue
            }
            message(experiment, 4, info.toByteArray());
        }
        for (String metric : metricDict.keySet()) {
            ByteArrayOutputStream mname = buf();
            string(mname, 2, metric); // MetricName.tag = 2
            ByteArrayOutputStream minfo = buf();
            message(minfo, 1, mname.toByteArray());
            message(experiment, 5, minfo.toByteArray());
        }

        ByteArrayOutputStream expPlugin = buf();
        int32(expPlugin, 1, 0); // version
        message(expPlugin, 2, experiment.toByteArray()); // experiment

        ByteArrayOutputStream expVal = buf();
        string(expVal, 1, "_hparams_/experiment");
        message(expVal, 9, pluginMetadata("hparams", expPlugin.toByteArray()));
        // empty tensor? pytorch still sets metadata only — actually no tensor needed;
        // TB only reads plugin_data.content. Provide empty float tensor for safety.
        message(expVal, 8, tensorProtoFloat(new long[0], new float[]{}));

        // ---- SessionStartInfo ----
        ByteArrayOutputStream ssi = buf();
        for (Map.Entry<String, ?> e : hparamDict.entrySet()) {
            if (e.getValue() == null) continue;
            // map<string, google.protobuf.Value> hparams = 1;
            // Map entry: key=1, value=2
            ByteArrayOutputStream entry = buf();
            string(entry, 1, e.getKey());
            message(entry, 2, structValue(e.getValue()));
            message(ssi, 1, entry.toByteArray());
        }
        ByteArrayOutputStream ssiPlugin = buf();
        int32(ssiPlugin, 1, 0);
        message(ssiPlugin, 3, ssi.toByteArray()); // session_start_info = field 3

        ByteArrayOutputStream ssiVal = buf();
        string(ssiVal, 1, "_hparams_/session_start_info");
        message(ssiVal, 9, pluginMetadata("hparams", ssiPlugin.toByteArray()));
        message(ssiVal, 8, tensorProtoFloat(new long[0], new float[]{}));

        // ---- SessionEndInfo STATUS_SUCCESS=1 ----
        ByteArrayOutputStream sei = buf();
        enum_(sei, 1, 1); // status
        ByteArrayOutputStream seiPlugin = buf();
        int32(seiPlugin, 1, 0);
        message(seiPlugin, 4, sei.toByteArray()); // session_end_info = field 4

        ByteArrayOutputStream seiVal = buf();
        string(seiVal, 1, "_hparams_/session_end_info");
        message(seiVal, 9, pluginMetadata("hparams", seiPlugin.toByteArray()));
        message(seiVal, 8, tensorProtoFloat(new long[0], new float[]{}));

        return new HparamsBundle(
                summaryOf(expVal.toByteArray()),
                summaryOf(ssiVal.toByteArray()),
                summaryOf(seiVal.toByteArray())
        );
    }

    public record HparamsBundle(byte[] experiment, byte[] sessionStart, byte[] sessionEnd) {}

    private static int hparamType(Object v) {
        if (v instanceof Boolean) return 2; // DATA_TYPE_BOOL
        if (v instanceof String) return 1;  // DATA_TYPE_STRING
        if (v instanceof Number) return 3;  // DATA_TYPE_FLOAT64
        throw new IllegalArgumentException("hparam value type not supported: " + v.getClass());
    }

    private static byte[] structValue(Object v) {
        if (v instanceof Boolean b) return structBool(b);
        if (v instanceof String s) return structString(s);
        if (v instanceof Number n) return structNumber(n.doubleValue());
        throw new IllegalArgumentException("unsupported struct value: " + v);
    }

    // ---- mesh ---------------------------------------------------------------

    /**
     * Mesh / point cloud. vertices/colors/faces shaped [B,N,3] flattened row-major.
     * components bitmask: VERTEX=1, FACE=2, COLOR=4 → OR together.
     */
    public static byte[] mesh(String tag,
                              float[] vertices, long[] vShape,
                              float[] faces, long[] fShape,
                              float[] colors, long[] cShape,
                              String jsonConfig) {
        int components = 0;
        if (vertices != null) components |= 1; // VERTEX
        if (faces != null) components |= 2;    // FACE  (bit 1 → value 2)
        if (colors != null) components |= 4;   // COLOR (bit 2 → value 4)
        // pytorch metadata.get_components_bitmask uses 1<<content_type:
        // VERTEX=1 → 2, FACE=2 → 4, COLOR=3 → 8 → OR = 14 when all three.
        // Recompute properly:
        components = 0;
        if (vertices != null) components |= (1 << 1); // VERTEX=1
        if (faces != null) components |= (1 << 2);    // FACE=2
        if (colors != null) components |= (1 << 3);   // COLOR=3

        if (jsonConfig == null) jsonConfig = "{}";
        List<byte[]> values = new ArrayList<>();
        if (vertices != null) {
            values.add(meshValue(tag, "VERTEX", 1, vertices, vShape, components, jsonConfig));
        }
        if (faces != null) {
            values.add(meshValue(tag, "FACE", 2, faces, fShape, components, jsonConfig));
        }
        if (colors != null) {
            values.add(meshValue(tag, "COLOR", 3, colors, cShape, components, jsonConfig));
        }
        ByteArrayOutputStream summary = buf();
        for (byte[] v : values) message(summary, 1, v);
        return summary.toByteArray();
    }

    private static byte[] meshValue(String baseTag, String suffix, int contentType,
                                    float[] data, long[] shape, int components, String jsonConfig) {
        // MeshPluginData: 1 version, 2 name, 3 content_type, 5 json_config, 6 shape (repeated int32), 7 components
        ByteArrayOutputStream pd = buf();
        int32(pd, 1, 0); // version
        string(pd, 2, baseTag);
        enum_(pd, 3, contentType);
        string(pd, 5, jsonConfig);
        if (shape != null) {
            for (long s : shape) int32(pd, 6, (int) s);
        }
        int32(pd, 7, components);

        ByteArrayOutputStream val = buf();
        string(val, 1, baseTag + "_" + suffix);
        message(val, 8, tensorProtoFloat(shape, data));
        message(val, 9, pluginMetadata("mesh", pd.toByteArray()));
        return val.toByteArray();
    }

    // ---- custom scalars layout ----------------------------------------------

    /**
     * {@code layout}: category → (chartName → (type, tags))
     * type is "Margin" or anything else (Multiline).
     * Margin tags must be length 3: value, lower, upper.
     */
    @SuppressWarnings("unchecked")
    public static byte[] customScalars(Map<String, Map<String, Object[]>> layout) {
        ByteArrayOutputStream layoutMsg = buf();
        // version = 1 (field 1)
        int32(layoutMsg, 1, 0);
        for (Map.Entry<String, Map<String, Object[]>> cat : layout.entrySet()) {
            ByteArrayOutputStream category = buf();
            string(category, 1, cat.getKey()); // title
            for (Map.Entry<String, Object[]> chart : cat.getValue().entrySet()) {
                Object[] meta = chart.getValue(); // [type, List<String> tags]
                String type = String.valueOf(meta[0]);
                List<String> tags = (List<String>) meta[1];
                ByteArrayOutputStream ch = buf();
                string(ch, 1, chart.getKey());
                if ("Margin".equalsIgnoreCase(type)) {
                    if (tags.size() != 3) throw new IllegalArgumentException("Margin chart needs 3 tags");
                    ByteArrayOutputStream series = buf();
                    string(series, 1, tags.get(0));
                    string(series, 2, tags.get(1));
                    string(series, 3, tags.get(2));
                    ByteArrayOutputStream margin = buf();
                    message(margin, 1, series.toByteArray());
                    message(ch, 3, margin.toByteArray()); // margin
                } else {
                    ByteArrayOutputStream ml = buf();
                    for (String t : tags) string(ml, 1, t);
                    message(ch, 2, ml.toByteArray()); // multiline
                }
                message(category, 2, ch.toByteArray());
            }
            message(layoutMsg, 2, category.toByteArray());
        }

        ByteArrayOutputStream val = buf();
        string(val, 1, "custom_scalars__config__");
        message(val, 8, tensorProtoStringBytes(new long[0], layoutMsg.toByteArray()));
        message(val, 9, pluginMetadata("custom_scalars"));
        return summaryOf(val.toByteArray());
    }

    // ---- helpers ------------------------------------------------------------

    /** Summary { repeated Value value = 1 } from one Value message. */
    public static byte[] summaryOf(byte[] valueMsg) {
        ByteArrayOutputStream s = buf();
        message(s, 1, valueMsg);
        return s.toByteArray();
    }

    public static byte[] summaryOfMany(Collection<byte[]> valueMsgs) {
        ByteArrayOutputStream s = buf();
        for (byte[] v : valueMsgs) message(s, 1, v);
        return s.toByteArray();
    }

    /** Encode mono/multi PCM float [-1,1] as 16-bit little-endian WAV. */
    public static byte[] encodeWavPcm16(float[] samples, int sampleRate, int channels) {
        int dataLen = samples.length * 2; // 16-bit
        ByteArrayOutputStream out = new ByteArrayOutputStream(44 + dataLen);
        try {
            out.write("RIFF".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(out, 36 + dataLen);
            out.write("WAVE".getBytes(StandardCharsets.US_ASCII));
            out.write("fmt ".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(out, 16); // PCM fmt chunk size
            writeShortLE(out, (short) 1); // PCM
            writeShortLE(out, (short) channels);
            writeIntLE(out, sampleRate);
            writeIntLE(out, sampleRate * channels * 2); // byte rate
            writeShortLE(out, (short) (channels * 2)); // block align
            writeShortLE(out, (short) 16); // bits
            out.write("data".getBytes(StandardCharsets.US_ASCII));
            writeIntLE(out, dataLen);
            for (float s : samples) {
                int v = Math.round(s * 32767f);
                if (v > 32767) v = 32767;
                if (v < -32768) v = -32768;
                writeShortLE(out, (short) v);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return out.toByteArray();
    }

    private static void writeIntLE(ByteArrayOutputStream out, int v) {
        out.write(v & 0xff);
        out.write((v >>> 8) & 0xff);
        out.write((v >>> 16) & 0xff);
        out.write((v >>> 24) & 0xff);
    }

    private static void writeShortLE(ByteArrayOutputStream out, short v) {
        out.write(v & 0xff);
        out.write((v >>> 8) & 0xff);
    }
}
