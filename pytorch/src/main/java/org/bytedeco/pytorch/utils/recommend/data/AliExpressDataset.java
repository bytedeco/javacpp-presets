/*
 * Ported from torch-rechub-scala: torchrec/data/AliExpressDataset.scala
 *
 * AliExpress multi-task (click + conversion) → MultiTaskDataset (native Dataset).
 * Local CSV under datasetPath, or synthetic multi-task fallback.
 *
 * Label encoding: 0→[0,0], 1→[1,0], 2→[1,1]  (or explicit click/conversion cols).
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
public final class AliExpressDataset {

    private static final int NUM_CATEGORICAL = 16;
    private static final int NUM_NUMERICAL = 40;

    private AliExpressDataset() {}

    public static final class Split {
        public final MultiTaskDataset train, test;
        public Split(MultiTaskDataset train, MultiTaskDataset test) {
            this.train = train;
            this.test = test;
        }
    }

    /**
     * Load from a directory containing train.csv / test.csv.
     * Falls back to synthetic if files missing.
     */
    public static Split load(String datasetPath) {
        return load(datasetPath, new String[]{"click", "conversion"});
    }

    public static Split load(String datasetPath, String[] taskNames) {
        System.out.println("============================================================");
        System.out.println("AliExpress Multi-Task Dataset Loading");
        System.out.println("============================================================");

        File dir = new File(datasetPath);
        File trainFile = new File(dir, "train.csv");
        File testFile = new File(dir, "test.csv");
        if (!trainFile.exists()) trainFile = new File(dir, "train.tsv");
        if (!testFile.exists()) testFile = new File(dir, "test.tsv");

        if (!trainFile.exists()) {
            System.out.println("  [Warn] " + trainFile + " missing — synthetic multi-task data.");
            return generateSynthetic(0.8f, 50_000, 42, taskNames);
        }

        System.out.println("  [Parse] " + trainFile.getAbsolutePath());
        MultiTaskDataset train = parseFile(trainFile, taskNames);
        MultiTaskDataset test;
        if (testFile.exists()) {
            System.out.println("  [Parse] " + testFile.getAbsolutePath());
            test = parseFile(testFile, taskNames);
        } else {
            // split train 80/20
            System.out.println("  [Warn] no test file — holding out 20% of train");
            long n = train.sizeLong();
            long trainN = (long) (n * 0.8);
            // rebuild via synthetic split of parsed tensors is heavy; re-parse with ratio
            test = parseFile(trainFile, taskNames); // same for now if no proper slice API on MultiTask
            // Prefer generating a proper holdout by re-reading
            Parsed both = parseRaw(trainFile);
            if (both.n > 0) {
                int tSize = (int) (both.n * 0.8);
                train = both.toDataset(0, tSize, taskNames);
                test = both.toDataset(tSize, both.n - tSize, taskNames);
            }
        }
        System.out.println("  [Dataset] train=" + train.sizeLong() + " test=" + test.sizeLong());
        System.out.println("============================================================");
        return new Split(train, test);
    }

    private static MultiTaskDataset parseFile(File file, String[] taskNames) {
        Parsed p = parseRaw(file);
        return p.toDataset(0, p.n, taskNames);
    }

    private static final class Parsed {
        float[][] cat;   // [NUM_CATEGORICAL][n]
        float[][] num;   // [NUM_NUMERICAL][n]
        float[] click;
        float[] conversion;
        int n;

        MultiTaskDataset toDataset(int start, int len, String[] taskNames) {
            if (len <= 0 || n == 0) {
                return generateSynthetic(1f, 100, 0, taskNames).train;
            }
            int end = Math.min(start + len, n);
            int m = end - start;
            Map<String, Tensor> features = new LinkedHashMap<>();
            for (int i = 0; i < NUM_CATEGORICAL; i++) {
                float[] col = new float[m];
                System.arraycopy(cat[i], start, col, 0, m);
                features.put("categorical_" + i,
                        RecommendDataset.floatFeature(col).toType(ScalarType.Long));
            }
            for (int i = 0; i < NUM_NUMERICAL; i++) {
                float[] col = new float[m];
                if (num[i] != null) System.arraycopy(num[i], start, col, 0, m);
                features.put("numerical_" + i, RecommendDataset.floatFeature(col));
            }
            Map<String, Tensor> tasks = new LinkedHashMap<>();
            String t0 = taskNames != null && taskNames.length > 0 ? taskNames[0] : "click";
            String t1 = taskNames != null && taskNames.length > 1 ? taskNames[1] : "conversion";
            float[] c0 = new float[m];
            float[] c1 = new float[m];
            System.arraycopy(click, start, c0, 0, m);
            System.arraycopy(conversion, start, c1, 0, m);
            tasks.put(t0, RecommendDataset.floatFeature(c0));
            tasks.put(t1, RecommendDataset.floatFeature(c1));
            return new MultiTaskDataset(features, tasks);
        }
    }

    private static Parsed parseRaw(File file) {
        List<float[]> catRows = new ArrayList<>();
        List<float[]> numRows = new ArrayList<>();
        List<Float> clicks = new ArrayList<>();
        List<Float> convs = new ArrayList<>();
        String delim = file.getName().endsWith(".tsv") ? "\t" : ",";

        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, delim, true, Long.MAX_VALUE)) {
            while (it.hasNext()) {
                String[] f = it.next();
                if (f.length < NUM_CATEGORICAL + 2) continue;
                float[] cats = new float[NUM_CATEGORICAL];
                for (int i = 0; i < NUM_CATEGORICAL; i++) {
                    cats[i] = parseFloatSafe(f[i], 0f);
                }
                float[] nums = new float[NUM_NUMERICAL];
                int numCount = Math.min(NUM_NUMERICAL, Math.max(0, f.length - NUM_CATEGORICAL - 2));
                for (int i = 0; i < numCount; i++) {
                    nums[i] = parseFloatSafe(f[NUM_CATEGORICAL + i], 0f);
                }
                // last two cols: click, conversion  OR single multi-class label
                float click, conv;
                if (f.length >= NUM_CATEGORICAL + numCount + 2) {
                    click = parseFloatSafe(f[f.length - 2], 0f);
                    conv = parseFloatSafe(f[f.length - 1], 0f);
                } else {
                    float lab = parseFloatSafe(f[f.length - 1], 0f);
                    // 0→[0,0] 1→[1,0] 2→[1,1]
                    click = lab >= 1f ? 1f : 0f;
                    conv = lab >= 2f ? 1f : 0f;
                }
                catRows.add(cats);
                numRows.add(nums);
                clicks.add(click);
                convs.add(conv);
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }

        Parsed p = new Parsed();
        p.n = catRows.size();
        p.cat = new float[NUM_CATEGORICAL][p.n];
        p.num = new float[NUM_NUMERICAL][p.n];
        p.click = new float[p.n];
        p.conversion = new float[p.n];
        for (int j = 0; j < p.n; j++) {
            for (int i = 0; i < NUM_CATEGORICAL; i++) p.cat[i][j] = catRows.get(j)[i];
            for (int i = 0; i < NUM_NUMERICAL; i++) p.num[i][j] = numRows.get(j)[i];
            p.click[j] = clicks.get(j);
            p.conversion[j] = convs.get(j);
        }
        return p;
    }

    private static float parseFloatSafe(String s, float dft) {
        if (s == null || s.isEmpty()) return dft;
        try {
            return Float.parseFloat(s.trim());
        } catch (NumberFormatException e) {
            return dft;
        }
    }

    public static Split generateSynthetic(float trainRatio, int numSamples, int seed, String[] taskNames) {
        Random rng = new Random(seed);
        int n = numSamples;
        Map<String, Tensor> features = new LinkedHashMap<>();
        for (int i = 0; i < NUM_CATEGORICAL; i++) {
            float[] col = new float[n];
            for (int j = 0; j < n; j++) col[j] = rng.nextInt(1000);
            features.put("categorical_" + i,
                    RecommendDataset.floatFeature(col).toType(ScalarType.Long));
        }
        for (int i = 0; i < NUM_NUMERICAL; i++) {
            float[] col = new float[n];
            for (int j = 0; j < n; j++) col[j] = rng.nextFloat();
            features.put("numerical_" + i, RecommendDataset.floatFeature(col));
        }
        float[] click = new float[n];
        float[] conv = new float[n];
        for (int j = 0; j < n; j++) {
            click[j] = rng.nextFloat() < 0.3f ? 1f : 0f;
            conv[j] = click[j] > 0 && rng.nextFloat() < 0.1f ? 1f : 0f;
        }
        String t0 = taskNames != null && taskNames.length > 0 ? taskNames[0] : "click";
        String t1 = taskNames != null && taskNames.length > 1 ? taskNames[1] : "conversion";
        Map<String, Tensor> tasks = new LinkedHashMap<>();
        tasks.put(t0, RecommendDataset.floatFeature(click));
        tasks.put(t1, RecommendDataset.floatFeature(conv));

        // Manual split by rebuilding feature maps narrowed — MultiTask has no slice;
        // build full then two datasets from index ranges via re-extract is complex.
        // Simpler: two independent synthetic sets with different seeds for test.
        MultiTaskDataset train = new MultiTaskDataset(features, tasks);
        // test: smaller independent draw
        int testN = Math.max(1000, n / 5);
        return new Split(train, generateSynthetic(1f, testN, seed + 7, taskNames).train);
    }

    public static int numCategorical() { return NUM_CATEGORICAL; }
    public static int numNumerical() { return NUM_NUMERICAL; }
}
