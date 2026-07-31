/*
 * Ported from torch-rechub-scala: torchrec/data/CriteoDataset.scala
 *
 * Criteo CTR ranking dataset. Produces TensorDataset (extends native Dataset).
 * Download → parse (13 dense + 26 sparse hash) → split → TensorDataset.
 * Falls back to synthetic data if download fails.
 */
package org.bytedeco.pytorch.recommend.data;

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
public final class CriteoDataset {

    private static final int NUM_DENSE = 13;
    private static final int NUM_SPARSE = 26;
    private static final int SPARSE_VOCAB = 100_000;
    private static final String DATASET_NAME = "criteo_day15";

    private CriteoDataset() {}

    public static final class Split {
        public final TensorDataset train;
        public final TensorDataset val;
        public final TensorDataset test;

        public Split(TensorDataset train, TensorDataset val, TensorDataset test) {
            this.train = train;
            this.val = val;
            this.test = test;
        }
    }

    public static Split load() {
        return load(0.8f, 1_000_000, 42);
    }

    public static Split load(float trainRatio, int maxSamples, int seed) {
        System.out.println("============================================================");
        System.out.println("Criteo CTR Dataset Loading");
        System.out.println("============================================================");

        File dataFile = tryDownload();
        if (dataFile == null) {
            System.out.println("  [Warn] Could not download Criteo. Generating synthetic data.");
            return generateSynthetic(trainRatio, maxSamples > 0 ? maxSamples : 100_000, seed);
        }

        System.out.println("  [Parse] Reading Criteo data...");
        List<CriteoRow> rows = parseCriteoFile(dataFile, maxSamples > 0 ? maxSamples : 1_000_000);
        System.out.println("  [Data] Loaded rows: " + rows.size());

        if (rows.isEmpty()) {
            System.out.println("  [Warn] No rows parsed. Falling back to synthetic.");
            return generateSynthetic(trainRatio, maxSamples > 0 ? maxSamples : 100_000, seed);
        }

        Random rng = new Random(seed);
        Collections.shuffle(rows, rng);

        int trainSize = (int) (rows.size() * trainRatio);
        int valSize = (rows.size() - trainSize) / 2;
        List<CriteoRow> trainRows = rows.subList(0, trainSize);
        List<CriteoRow> valRows = rows.subList(trainSize, trainSize + valSize);
        List<CriteoRow> testRows = rows.subList(trainSize + valSize, rows.size());

        System.out.println("  [Split] Train: " + trainRows.size()
                + ", Val: " + valRows.size() + ", Test: " + testRows.size());

        TensorDataset trainDS = buildDataset(trainRows);
        TensorDataset valDS = buildDataset(valRows);
        TensorDataset testDS = buildDataset(testRows);

        long pos = 0;
        for (CriteoRow r : rows) if (r.label > 0.5f) pos++;
        System.out.printf("  [Data] Positive rate: %.2f%%%n", pos * 100.0 / rows.size());
        System.out.println("============================================================");

        return new Split(trainDS, valDS, testDS);
    }

    private static File tryDownload() {
        String[][] urls = {
                {"https://raw.githubusercontent.com/mkechinov/criteo-analysis/master/data/day_15_roc", "criteo_day15"},
                {"https://raw.githubusercontent.com/makefu/criteo-analysis/master/data/day_15_roc", "criteo_day15_alt"},
        };
        for (String[] pair : urls) {
            try {
                System.out.println("  [Try] " + pair[0]);
                File file = DatasetDownloader.download(pair[0], pair[1], false);
                if (file.exists() && file.length() > 1000) {
                    System.out.println("  [OK] Downloaded: " + file.length() + " bytes");
                    return file;
                }
            } catch (Throwable t) {
                System.out.println("  [Fail] " + t.getMessage());
            }
        }
        return null;
    }

    private static final class CriteoRow {
        final float label;
        final float[] dense;
        final float[] sparse;

        CriteoRow(float label, float[] dense, float[] sparse) {
            this.label = label;
            this.dense = dense;
            this.sparse = sparse;
        }
    }

    private static List<CriteoRow> parseCriteoFile(File file, int maxRows) {
        List<CriteoRow> out = new ArrayList<>();
        try (DatasetDownloader.LineIterator lines =
                     DatasetDownloader.readLines(file, "\t", false, maxRows)) {
            int count = 0;
            while (lines.hasNext()) {
                if (count >= maxRows) break;
                CriteoRow row = parseCriteoRow(lines.next());
                if (row != null) {
                    out.add(row);
                    count++;
                    if (count % 100_000 == 0) {
                        System.out.println("    Parsed " + count + " rows...");
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }
        return out;
    }

    private static CriteoRow parseCriteoRow(String[] fields) {
        if (fields == null || fields.length < 40) return null;
        try {
            float label = Float.parseFloat(fields[0]);
            float[] dense = new float[NUM_DENSE];
            for (int i = 0; i < NUM_DENSE; i++) {
                String raw = fields[i + 1];
                if (raw == null || raw.isEmpty()) {
                    dense[i] = 0f;
                } else {
                    try {
                        dense[i] = Float.parseFloat(raw);
                    } catch (NumberFormatException e) {
                        dense[i] = 0f;
                    }
                }
            }
            float[] sparse = new float[NUM_SPARSE];
            for (int i = 0; i < NUM_SPARSE; i++) {
                int idx = 1 + NUM_DENSE + i;
                String raw = idx < fields.length ? fields[idx] : "";
                sparse[i] = Math.floorMod(hashString(raw), SPARSE_VOCAB);
            }
            return new CriteoRow(label, dense, sparse);
        } catch (Throwable t) {
            return null;
        }
    }

    private static int hashString(String s) {
        if (s == null || s.isEmpty()) return 0;
        int h = 0;
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return h;
    }

    private static TensorDataset buildDataset(List<CriteoRow> rows) {
        int n = rows.size();
        float[][] denseArrays = new float[NUM_DENSE][n];
        float[][] sparseArrays = new float[NUM_SPARSE][n];
        float[] labels = new float[n];
        for (int j = 0; j < n; j++) {
            CriteoRow row = rows.get(j);
            for (int i = 0; i < NUM_DENSE; i++) denseArrays[i][j] = row.dense[i];
            for (int i = 0; i < NUM_SPARSE; i++) sparseArrays[i][j] = row.sparse[i];
            labels[j] = row.label;
        }

        Map<String, Tensor> sparseFeatures = new LinkedHashMap<>();
        for (int i = 0; i < NUM_SPARSE; i++) {
            sparseFeatures.put("sparse_" + i,
                    RecommendDataset.floatFeature(sparseArrays[i]).toType(ScalarType.Long));
        }
        Map<String, Tensor> denseFeatures = new LinkedHashMap<>();
        for (int i = 0; i < NUM_DENSE; i++) {
            denseFeatures.put("dense_" + i, RecommendDataset.floatFeature(denseArrays[i]));
        }
        Tensor labelsTensor = RecommendDataset.floatFeature(labels);
        return new TensorDataset(sparseFeatures, denseFeatures, labelsTensor);
    }

    public static Split generateSynthetic(float trainRatio, int numSamples, int seed) {
        Random rng = new Random(seed);
        int n = numSamples;
        float[][] denseArrays = new float[NUM_DENSE][n];
        float[][] sparseArrays = new float[NUM_SPARSE][n];
        float[] labels = new float[n];

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < NUM_DENSE; i++) denseArrays[i][j] = rng.nextFloat() * 100f;
            for (int i = 0; i < NUM_SPARSE; i++) sparseArrays[i][j] = rng.nextInt(10_000);
            double score = 0;
            for (int i = 0; i < 3; i++) score += (denseArrays[i][j] / 100.0) * (i + 1);
            for (int i = 0; i < 5; i++) score += (sparseArrays[i][j] / 10000.0) * (i + 1);
            float prob = (float) (1.0 / (1.0 + Math.exp(-score)));
            labels[j] = rng.nextFloat() < prob ? 1f : 0f;
        }

        Map<String, Tensor> sparseFeatures = new LinkedHashMap<>();
        for (int i = 0; i < NUM_SPARSE; i++) {
            sparseFeatures.put("sparse_" + i,
                    RecommendDataset.floatFeature(sparseArrays[i]).toType(ScalarType.Long));
        }
        Map<String, Tensor> denseFeatures = new LinkedHashMap<>();
        for (int i = 0; i < NUM_DENSE; i++) {
            denseFeatures.put("dense_" + i, RecommendDataset.floatFeature(denseArrays[i]));
        }
        Tensor labelsTensor = RecommendDataset.floatFeature(labels);

        int trainSize = (int) (n * trainRatio);
        int valSize = (n - trainSize) / 2;
        int testSize = n - trainSize - valSize;

        TensorDataset full = new TensorDataset(sparseFeatures, denseFeatures, labelsTensor);
        return new Split(
                full.slice(0, trainSize),
                full.slice(trainSize, valSize),
                full.slice(trainSize + valSize, testSize));
    }

    public static int numDense() { return NUM_DENSE; }
    public static int numSparse() { return NUM_SPARSE; }
    public static int sparseVocab() { return SPARSE_VOCAB; }
}
