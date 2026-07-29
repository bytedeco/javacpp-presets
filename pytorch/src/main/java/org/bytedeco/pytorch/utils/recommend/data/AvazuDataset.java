/*
 * Ported from torch-rechub-scala: torchrec/data/AvazuDataset.scala
 *
 * Avazu CTR ranking dataset → TensorDataset (native Dataset).
 * 22 hashed categorical features + click label. Synthetic fallback.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class AvazuDataset {

    private static final int NUM_FEATURES = 22;
    private static final int VOCAB = 100_000;

    private AvazuDataset() {}

    public static final class Split {
        public final TensorDataset train, val, test;
        public Split(TensorDataset train, TensorDataset val, TensorDataset test) {
            this.train = train; this.val = val; this.test = test;
        }
    }

    public static Split load() { return load(0.8f, 100_000, 42); }

    public static Split load(float trainRatio, int maxSamples, int seed) {
        System.out.println("============================================================");
        System.out.println("Avazu CTR Dataset Loading");
        System.out.println("============================================================");

        File dataFile = tryDownload();
        if (dataFile == null || !dataFile.exists() || dataFile.length() < 1000) {
            System.out.println("  [Warn] Could not download Avazu. Generating synthetic data.");
            return generateSynthetic(trainRatio, maxSamples > 0 ? maxSamples : 100_000, seed);
        }

        System.out.println("  [Parse] Reading Avazu data...");
        List<Row> rows = parseFile(dataFile, maxSamples > 0 ? maxSamples : 10_000_000);
        System.out.println("  [Data] Loaded rows: " + rows.size());
        if (rows.isEmpty()) {
            return generateSynthetic(trainRatio, maxSamples > 0 ? maxSamples : 100_000, seed);
        }

        Random rng = new Random(seed);
        Collections.shuffle(rows, rng);
        int trainSize = (int) (rows.size() * trainRatio);
        int valSize = (rows.size() - trainSize) / 2;
        TensorDataset train = build(rows.subList(0, trainSize));
        TensorDataset val = build(rows.subList(trainSize, trainSize + valSize));
        TensorDataset test = build(rows.subList(trainSize + valSize, rows.size()));
        long pos = 0;
        for (Row r : rows) if (r.label > 0.5f) pos++;
        System.out.printf("  [Split] Train/Val/Test = %d/%d/%d  pos=%.2f%%%n",
                train.sizeLong(), val.sizeLong(), test.sizeLong(), pos * 100.0 / rows.size());
        System.out.println("============================================================");
        return new Split(train, val, test);
    }

    private static File tryDownload() {
        String[] urls = {
                "https://raw.githubusercontent.com/Avazu/azureML/master/avazu_ctr/avazu_data",
        };
        for (String url : urls) {
            try {
                System.out.println("  [Try] " + url);
                File f = DatasetDownloader.download(url, "avazu", false);
                if (f.exists() && f.length() > 10_000) return f;
            } catch (Throwable t) {
                System.out.println("  [Fail] " + t.getMessage());
            }
        }
        return null;
    }

    private static final class Row {
        final float label;
        final float[] features;
        Row(float label, float[] features) { this.label = label; this.features = features; }
    }

    private static List<Row> parseFile(File file, int maxRows) {
        List<Row> out = new ArrayList<>();
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, ",", true, maxRows)) {
            while (it.hasNext() && out.size() < maxRows) {
                Row r = parseRow(it.next());
                if (r != null) {
                    out.add(r);
                    if (out.size() % 100_000 == 0)
                        System.out.println("    Parsed " + out.size() + " rows...");
                }
            }
        } catch (Exception e) {
            System.out.println("  [Parse Error] " + e.getMessage());
        }
        return out;
    }

    private static Row parseRow(String[] fields) {
        if (fields == null || fields.length < NUM_FEATURES + 1) return null;
        try {
            float label = Float.parseFloat(fields[0]);
            float[] feats = new float[NUM_FEATURES];
            for (int i = 0; i < NUM_FEATURES; i++) {
                int idx = i + 1;
                String raw = idx < fields.length ? fields[idx] : "";
                feats[i] = Math.floorMod(hash(raw), VOCAB);
            }
            return new Row(label, feats);
        } catch (Throwable t) {
            return null;
        }
    }

    private static int hash(String s) {
        if (s == null || s.isEmpty()) return 0;
        int h = 0;
        for (int i = 0; i < s.length(); i++) h = 31 * h + s.charAt(i);
        return h;
    }

    private static TensorDataset build(List<Row> rows) {
        int n = rows.size();
        float[][] feat = new float[NUM_FEATURES][n];
        float[] labels = new float[n];
        for (int j = 0; j < n; j++) {
            Row r = rows.get(j);
            for (int i = 0; i < NUM_FEATURES; i++) feat[i][j] = r.features[i];
            labels[j] = r.label;
        }
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        for (int i = 0; i < NUM_FEATURES; i++) {
            sparse.put("feat_" + i, RecommendDataset.floatFeature(feat[i]).toType(ScalarType.Long));
        }
        return new TensorDataset(sparse, Collections.emptyMap(), RecommendDataset.floatFeature(labels));
    }

    public static Split generateSynthetic(float trainRatio, int numSamples, int seed) {
        Random rng = new Random(seed);
        int n = numSamples;
        float[][] feat = new float[NUM_FEATURES][n];
        float[] labels = new float[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < NUM_FEATURES; i++) feat[i][j] = rng.nextInt(50_000);
            double score = 0;
            for (int i = 0; i < 5; i++) score += (feat[i][j] / 50000.0) * (i + 1) * 0.5;
            float prob = (float) (1.0 / (1.0 + Math.exp(-score)));
            labels[j] = rng.nextFloat() < prob ? 1f : 0f;
        }
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        for (int i = 0; i < NUM_FEATURES; i++) {
            sparse.put("feat_" + i, RecommendDataset.floatFeature(feat[i]).toType(ScalarType.Long));
        }
        TensorDataset full = new TensorDataset(sparse, Collections.emptyMap(),
                RecommendDataset.floatFeature(labels));
        int trainSize = (int) (n * trainRatio);
        int valSize = (n - trainSize) / 2;
        return new Split(full.slice(0, trainSize), full.slice(trainSize, valSize),
                full.slice(trainSize + valSize, n - trainSize - valSize));
    }

    public static int numFeatures() { return NUM_FEATURES; }
}
