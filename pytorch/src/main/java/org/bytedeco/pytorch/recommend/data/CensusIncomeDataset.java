/*
 * Ported from torch-rechub-scala: torchrec/data/CensusIncomeDataset.scala
 *
 * UCI Adult / Census-Income binary classification → TensorDataset (native Dataset).
 * 6 continuous + 8 categorical (label-encoded) features.
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CensusIncomeDataset {

    private static final String TRAIN_URL =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data";
    private static final String TEST_URL =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test";

    // indices: 0 age(c), 1 workclass(cat), 2 fnlwgt(c), 3 education(cat), 4 education-num(c),
    // 5 marital(cat), 6 occupation(cat), 7 relationship(cat), 8 race(cat), 9 sex(cat),
    // 10 capital-gain(c), 11 capital-loss(c), 12 hours(c), 13 native-country(cat), 14 label
    private static final int[] CONT_IDX = {0, 2, 4, 10, 11, 12};
    private static final int[] CAT_IDX = {1, 3, 5, 6, 7, 8, 9, 13};

    private CensusIncomeDataset() {}

    public static final class Split {
        public final TensorDataset train, val, test;
        public Split(TensorDataset train, TensorDataset val, TensorDataset test) {
            this.train = train; this.val = val; this.test = test;
        }
    }

    public static Split load() { return load(0.8f, false, 42); }

    public static Split load(float trainRatio, boolean useOfficialTest, int seed) {
        System.out.println("============================================================");
        System.out.println("Census-Income Dataset Loading");
        System.out.println("============================================================");

        File trainFile;
        try {
            System.out.println("  [Download] Training data...");
            trainFile = DatasetDownloader.download(TRAIN_URL, "census_income_train");
        } catch (Throwable t) {
            System.out.println("  [Warn] Download failed: " + t.getMessage() + " — synthetic fallback.");
            return generateSynthetic(trainRatio, 30_000, seed);
        }

        List<RawRow> trainRows = parseAdultFile(trainFile);
        System.out.println("  [Data] Training rows: " + trainRows.size());
        if (trainRows.isEmpty()) {
            return generateSynthetic(trainRatio, 30_000, seed);
        }

        List<RawRow> testRows = Collections.emptyList();
        if (useOfficialTest) {
            try {
                File testFile = DatasetDownloader.download(TEST_URL, "census_income_test");
                testRows = parseAdultFile(testFile);
                System.out.println("  [Data] Test rows: " + testRows.size());
            } catch (Throwable t) {
                System.out.println("  [Warn] Official test download failed: " + t.getMessage());
            }
        }

        // Fit category encoders on all available rows
        Map<Integer, Map<String, Integer>> encoders = fitEncoders(trainRows, testRows);

        Random rng = new Random(seed);
        if (useOfficialTest && !testRows.isEmpty()) {
            Collections.shuffle(trainRows, rng);
            int trainSize = (int) (trainRows.size() * trainRatio);
            TensorDataset train = build(trainRows.subList(0, trainSize), encoders);
            TensorDataset val = build(trainRows.subList(trainSize, trainRows.size()), encoders);
            TensorDataset test = build(testRows, encoders);
            System.out.println("============================================================");
            return new Split(train, val, test);
        }

        Collections.shuffle(trainRows, rng);
        int trainSize = (int) (trainRows.size() * trainRatio);
        int valSize = (trainRows.size() - trainSize) / 2;
        TensorDataset train = build(trainRows.subList(0, trainSize), encoders);
        TensorDataset val = build(trainRows.subList(trainSize, trainSize + valSize), encoders);
        TensorDataset test = build(trainRows.subList(trainSize + valSize, trainRows.size()), encoders);
        System.out.println("  [Split] " + train.sizeLong() + "/" + val.sizeLong() + "/" + test.sizeLong());
        System.out.println("============================================================");
        return new Split(train, val, test);
    }

    private static final class RawRow {
        final String[] fields;
        final float label;
        RawRow(String[] fields, float label) { this.fields = fields; this.label = label; }
    }

    private static List<RawRow> parseAdultFile(File file) {
        List<RawRow> out = new ArrayList<>();
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, ",", false, Long.MAX_VALUE)) {
            while (it.hasNext()) {
                String[] raw = it.next();
                // trim fields
                String[] fields = new String[raw.length];
                for (int i = 0; i < raw.length; i++) {
                    fields[i] = raw[i] == null ? "" : raw[i].trim();
                }
                if (fields.length < 15) continue;
                // skip empty
                if (fields[0].isEmpty()) continue;
                String lab = fields[14].replace(".", ""); // test file has trailing dot
                float y = (lab.contains(">50K")) ? 1f : 0f;
                out.add(new RawRow(fields, y));
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }
        return out;
    }

    private static Map<Integer, Map<String, Integer>> fitEncoders(List<RawRow> a, List<RawRow> b) {
        Map<Integer, Map<String, Integer>> enc = new HashMap<>();
        for (int cat : CAT_IDX) enc.put(cat, new HashMap<>());
        for (RawRow r : a) observe(enc, r);
        for (RawRow r : b) observe(enc, r);
        return enc;
    }

    private static void observe(Map<Integer, Map<String, Integer>> enc, RawRow r) {
        for (int cat : CAT_IDX) {
            if (cat >= r.fields.length) continue;
            String v = r.fields[cat];
            if (v == null || v.isEmpty() || "?".equals(v)) v = "UNK";
            Map<String, Integer> m = enc.get(cat);
            if (!m.containsKey(v)) m.put(v, m.size() + 1); // 0 reserved for UNK/missing
        }
    }

    private static TensorDataset build(List<RawRow> rows, Map<Integer, Map<String, Integer>> enc) {
        int n = rows.size();
        Map<String, Tensor> dense = new LinkedHashMap<>();
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        float[] labels = new float[n];

        for (int c = 0; c < CONT_IDX.length; c++) {
            float[] col = new float[n];
            int src = CONT_IDX[c];
            for (int j = 0; j < n; j++) {
                try {
                    col[j] = Float.parseFloat(rows.get(j).fields[src]);
                } catch (Throwable t) {
                    col[j] = 0f;
                }
            }
            dense.put("cont_" + c, RecommendDataset.floatFeature(col));
        }
        for (int c = 0; c < CAT_IDX.length; c++) {
            float[] col = new float[n];
            int src = CAT_IDX[c];
            Map<String, Integer> m = enc.get(src);
            for (int j = 0; j < n; j++) {
                String v = rows.get(j).fields[src];
                if (v == null || v.isEmpty() || "?".equals(v)) v = "UNK";
                col[j] = m.getOrDefault(v, 0);
            }
            sparse.put("cat_" + c, RecommendDataset.floatFeature(col).toType(ScalarType.Long));
        }
        for (int j = 0; j < n; j++) labels[j] = rows.get(j).label;
        return new TensorDataset(sparse, dense, RecommendDataset.floatFeature(labels));
    }

    public static Split generateSynthetic(float trainRatio, int numSamples, int seed) {
        Random rng = new Random(seed);
        int n = numSamples;
        Map<String, Tensor> dense = new LinkedHashMap<>();
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        float[] labels = new float[n];
        for (int c = 0; c < CONT_IDX.length; c++) {
            float[] col = new float[n];
            for (int j = 0; j < n; j++) col[j] = rng.nextFloat() * 100f;
            dense.put("cont_" + c, RecommendDataset.floatFeature(col));
        }
        for (int c = 0; c < CAT_IDX.length; c++) {
            float[] col = new float[n];
            for (int j = 0; j < n; j++) col[j] = rng.nextInt(20);
            sparse.put("cat_" + c, RecommendDataset.floatFeature(col).toType(ScalarType.Long));
        }
        for (int j = 0; j < n; j++) labels[j] = rng.nextFloat() < 0.24f ? 1f : 0f;
        TensorDataset full = new TensorDataset(sparse, dense, RecommendDataset.floatFeature(labels));
        int trainSize = (int) (n * trainRatio);
        int valSize = (n - trainSize) / 2;
        return new Split(full.slice(0, trainSize), full.slice(trainSize, valSize),
                full.slice(trainSize + valSize, n - trainSize - valSize));
    }
}
