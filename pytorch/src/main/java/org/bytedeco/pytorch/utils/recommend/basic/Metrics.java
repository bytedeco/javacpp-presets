/*
 * Ported from torch-rechub-scala: torchrec/basic/Metrics.scala
 *
 * Evaluation metrics for recommender systems.
 * Accuracy: AUC, GAUC, LogLoss
 * Top-K: NDCG, MRR, Recall, Hit, Precision
 * Beyond-accuracy: Diversity, Coverage, Novelty
 */
package org.bytedeco.pytorch.utils.recommend.basic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class Metrics {

    private Metrics() {}

    /** Compute AUC score. */
    public static float aucScore(float[] yTrue, float[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("y_true and y_pred must have same length");
        }
        int n = yTrue.length;
        if (n == 0) return 0.0f;

        long posCount = 0L;
        long negCount = 0L;
        long tieCount = 0L;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float ti = yTrue[i];
                float tj = yTrue[j];
                float pi = yPred[i];
                float pj = yPred[j];

                if (ti > tj) {
                    if (pi > pj) posCount += 1;
                    else if (pi < pj) negCount += 1;
                    else tieCount += 1;
                } else if (ti < tj) {
                    if (pi < pj) posCount += 1;
                    else if (pi > pj) negCount += 1;
                    else tieCount += 1;
                }
            }
        }

        if (posCount + negCount == 0) return 0.5f;
        return (posCount + 0.5f * tieCount) / (float) (posCount + negCount);
    }

    private static final class UserPred {
        final List<Float> yTrue = new ArrayList<>();
        final List<Float> yPred = new ArrayList<>();
    }

    private static Map<Integer, UserPred> getUserPred(float[] yTrue, float[] yPred, int[] users) {
        Map<Integer, UserPred> userPred = new HashMap<>();
        for (int i = 0; i < yTrue.length; i++) {
            int u = users[i];
            UserPred pred = userPred.get(u);
            if (pred == null) {
                pred = new UserPred();
                userPred.put(u, pred);
            }
            pred.yTrue.add(yTrue[i]);
            pred.yPred.add(yPred[i]);
        }
        return userPred;
    }

    /** Compute Group-AUC (GAUC). */
    public static float gaucScore(float[] yTrue, float[] yPred, int[] users) {
        return gaucScore(yTrue, yPred, users, null);
    }

    public static float gaucScore(float[] yTrue, float[] yPred, int[] users, Map<Integer, Float> weights) {
        if (yTrue.length != yPred.length || yTrue.length != users.length) {
            throw new IllegalArgumentException("y_true, y_pred, and users must have same length");
        }

        Map<Integer, UserPred> userPred = getUserPred(yTrue, yPred, users);
        float score = 0.0f;
        float num = 0.0f;

        for (Map.Entry<Integer, UserPred> e : userPred.entrySet()) {
            int u = e.getKey();
            UserPred pred = e.getValue();
            float[] yt = toFloatArray(pred.yTrue);
            float[] yp = toFloatArray(pred.yPred);
            float auc = aucScore(yt, yp);
            float userWeight;
            if (weights != null && weights.containsKey(u)) {
                userWeight = weights.get(u);
            } else {
                userWeight = pred.yTrue.size();
            }
            score += auc * userWeight;
            num += userWeight;
        }

        return num == 0 ? 0.0f : score / num;
    }

    public static Map<String, Float> ndcgScore(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        Map<String, String> result = topkMetrics(yTrue, yPred, ks);
        Map<String, Float> out = new LinkedHashMap<>();
        for (int k : ks) {
            out.put("NDCG@" + k, Float.parseFloat(result.get("NDCG@" + k)));
        }
        return out;
    }

    public static Map<String, Float> hitScore(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        Map<String, String> result = topkMetrics(yTrue, yPred, ks);
        Map<String, Float> out = new LinkedHashMap<>();
        for (int k : ks) {
            out.put("Hits@" + k, Float.parseFloat(result.get("Hits@" + k)));
        }
        return out;
    }

    public static Map<String, Float> mrrScore(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        Map<String, String> result = topkMetrics(yTrue, yPred, ks);
        Map<String, Float> out = new LinkedHashMap<>();
        for (int k : ks) {
            out.put("MRR@" + k, Float.parseFloat(result.get("MRR@" + k)));
        }
        return out;
    }

    public static Map<String, Float> recallScore(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        Map<String, String> result = topkMetrics(yTrue, yPred, ks);
        Map<String, Float> out = new LinkedHashMap<>();
        for (int k : ks) {
            out.put("Recall@" + k, Float.parseFloat(result.get("Recall@" + k)));
        }
        return out;
    }

    public static Map<String, Float> precisionScore(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        Map<String, String> result = topkMetrics(yTrue, yPred, ks);
        Map<String, Float> out = new LinkedHashMap<>();
        for (int k : ks) {
            out.put("Precision@" + k, Float.parseFloat(result.get("Precision@" + k)));
        }
        return out;
    }

    /** Compute top-K metrics: NDCG, MRR, Recall, Hit, Precision. */
    public static Map<String, String> topkMetrics(
            Map<String, List<Integer>> yTrue,
            Map<String, List<Integer>> yPred,
            int... topKs) {
        if (yTrue.size() != yPred.size()) {
            throw new IllegalArgumentException("y_true and y_pred must have same size");
        }
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        if (ks.length == 0) {
            throw new IllegalArgumentException("topKs must not be empty");
        }

        Map<String, String> results = new LinkedHashMap<>();
        List<String> userIds = new ArrayList<>(yTrue.keySet());
        int numUsers = userIds.size();

        for (int k : ks) {
            double ndcgs = 0.0;
            double mrrs = 0.0;
            double hits = 0.0;
            double precisions = 0.0;
            double recalls = 0.0;
            long gts = 0L;

            for (String u : userIds) {
                List<Integer> trueItems = yTrue.get(u);
                List<Integer> predItems = yPred.get(u);
                if (trueItems == null || trueItems.isEmpty()) continue;

                double mrrTmp = 0.0;
                boolean mrrFlag = true;
                double hitTmp = 0.0;
                double dcgTmp = 0.0;
                double idcgTmp = 0.0;

                int limit = Math.min(k, predItems != null ? predItems.size() : 0);
                for (int j = 0; j < limit; j++) {
                    if (trueItems.contains(predItems.get(j))) {
                        hitTmp += 1.0;
                        if (mrrFlag) {
                            mrrFlag = false;
                            mrrTmp = 1.0 / (1 + j);
                        }
                        dcgTmp += 1.0 / (Math.log(j + 2) / Math.log(2));
                    }
                    if (j < trueItems.size()) {
                        idcgTmp += 1.0 / (Math.log(j + 2) / Math.log(2));
                    }
                }

                gts += trueItems.size();
                hits += hitTmp;
                mrrs += mrrTmp;
                recalls += hitTmp / trueItems.size();
                precisions += hitTmp / k;
                if (idcgTmp != 0) {
                    ndcgs += dcgTmp / idcgTmp;
                }
            }

            double ndcgVal = roundDecimals(ndcgs / numUsers, 4);
            double mrrVal = roundDecimals(mrrs / numUsers, 4);
            double recallVal = roundDecimals(recalls / numUsers, 4);
            double hitVal = gts == 0 ? 0.0 : roundDecimals(hits / gts, 4);
            double precisionVal = roundDecimals(precisions / numUsers, 4);

            results.put("NDCG@" + k, String.format("%.4f", ndcgVal));
            results.put("MRR@" + k, String.format("%.4f", mrrVal));
            results.put("Recall@" + k, String.format("%.4f", recallVal));
            results.put("Hits@" + k, String.format("%.4f", hitVal));
            results.put("Precision@" + k, String.format("%.4f", precisionVal));
        }

        return results;
    }

    /**
     * Compute log loss.
     * Note: Scala formula is {@code (-total / count / 2)} — ported as-is.
     */
    public static float logLoss(float[] yTrue, float[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("y_true and y_pred must have same length");
        }
        float eps = 1e-10f;
        double total = 0.0;
        int count = yTrue.length;
        for (int i = 0; i < count; i++) {
            float yt = yTrue[i];
            float yp = yPred[i];
            double p = Math.max(eps, Math.min(1.0f - eps, yp));
            double lossItem = yt * Math.log(p) + (1.0 - yt) * Math.log(1.0 - p);
            total += lossItem;
        }
        return (float) (-total / count / 2);
    }

    /** Intra-List Diversity (ILD). */
    public static Map<String, String> diversityScore(
            Map<String, List<Integer>> yPred,
            Map<Integer, float[]> itemEmbeddings,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        if (ks.length == 0) {
            throw new IllegalArgumentException("topKs must not be empty");
        }

        Map<String, String> results = new LinkedHashMap<>();
        for (int k : ks) {
            List<Double> userDiversities = new ArrayList<>();

            for (List<Integer> items : yPred.values()) {
                List<Integer> selected = items.subList(0, Math.min(k, items.size()));
                if (selected.size() < 2) continue;

                List<float[]> embs = new ArrayList<>();
                for (int item : selected) {
                    float[] emb = itemEmbeddings.get(item);
                    if (emb != null) embs.add(emb);
                }
                if (embs.size() < 2) continue;

                int n = embs.size();
                double[] norms = new double[n];
                for (int i = 0; i < n; i++) {
                    double sum = 0.0;
                    for (float v : embs.get(i)) sum += (double) v * v;
                    norms[i] = Math.max(Math.sqrt(sum), 1e-10);
                }

                float[][] normed = new float[n][];
                for (int i = 0; i < n; i++) {
                    float[] emb = embs.get(i);
                    float[] row = new float[emb.length];
                    for (int d = 0; d < emb.length; d++) {
                        row[d] = (float) (emb[d] / norms[i]);
                    }
                    normed[i] = row;
                }

                double distSum = 0.0;
                int pairCount = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        double sim = 0.0;
                        for (int d = 0; d < normed[0].length; d++) {
                            sim += normed[i][d] * normed[j][d];
                        }
                        distSum += 1.0 - sim;
                        pairCount += 1;
                    }
                }
                if (pairCount > 0) {
                    userDiversities.add(distSum / pairCount);
                }
            }

            double score;
            if (!userDiversities.isEmpty()) {
                double sum = 0.0;
                for (double v : userDiversities) sum += v;
                score = roundDecimals(sum / userDiversities.size(), 4);
            } else {
                score = 0.0;
            }
            results.put("Diversity@" + k, String.format("%.4f", score));
        }
        return results;
    }

    /** Catalog Coverage. */
    public static Map<String, String> coverageScore(
            Map<String, List<Integer>> yPred,
            Set<Integer> allItems,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        if (ks.length == 0) {
            throw new IllegalArgumentException("topKs must not be empty");
        }
        if (allItems == null || allItems.isEmpty()) {
            throw new IllegalArgumentException("allItems must not be empty");
        }

        Map<String, String> results = new LinkedHashMap<>();
        for (int k : ks) {
            Set<Integer> recItems = new HashSet<>();
            for (List<Integer> items : yPred.values()) {
                int limit = Math.min(k, items.size());
                for (int i = 0; i < limit; i++) {
                    recItems.add(items.get(i));
                }
            }
            double score = roundDecimals((double) recItems.size() / allItems.size(), 4);
            results.put("Coverage@" + k, String.format("%.4f", score));
        }
        return results;
    }

    /** Mean Self-Information (Novelty). */
    public static Map<String, String> noveltyScore(
            Map<String, List<Integer>> yPred,
            Map<Integer, Float> itemPopularity,
            int... topKs) {
        int[] ks = topKs.length == 0 ? new int[]{5} : topKs;
        if (ks.length == 0) {
            throw new IllegalArgumentException("topKs must not be empty");
        }

        Map<String, String> results = new LinkedHashMap<>();
        for (int k : ks) {
            List<Double> userNovelties = new ArrayList<>();
            for (List<Integer> items : yPred.values()) {
                List<Integer> selected = items.subList(0, Math.min(k, items.size()));
                if (selected.isEmpty()) continue;
                double selfInfoSum = 0.0;
                for (int item : selected) {
                    Float p = itemPopularity.get(item);
                    float pop = p != null ? Math.max(p, 1e-10f) : 1e-10f;
                    selfInfoSum += -(Math.log(pop) / Math.log(2));
                }
                userNovelties.add(selfInfoSum / selected.size());
            }

            double score;
            if (!userNovelties.isEmpty()) {
                double sum = 0.0;
                for (double v : userNovelties) sum += v;
                score = roundDecimals(sum / userNovelties.size(), 2);
            } else {
                score = 0.0;
            }
            results.put("Novelty@" + k, String.format("%.2f", score));
        }
        return results;
    }

    private static float[] toFloatArray(List<Float> list) {
        float[] a = new float[list.size()];
        for (int i = 0; i < list.size(); i++) a[i] = list.get(i);
        return a;
    }

    private static double roundDecimals(double x, int decimals) {
        double factor = Math.pow(10, decimals);
        return Math.round(x * factor) / factor;
    }
}
