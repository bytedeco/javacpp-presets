package org.bytedeco.pytorch.data.dataframe.ml.metrics;

import java.util.*;

/**
 * 分类评估指标（对应 sklearn.metrics 分类部分）
 * 全部为静态方法，可直接调用
 */
public class ClassificationMetrics {

    // ============================================================
    // 准确率
    // ============================================================

    public static double accuracyScore(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) if (yTrue[i] == yPred[i]) correct++;
        return (double) correct / yTrue.length;
    }

    // ============================================================
    // 精确率 / 召回率 / F1（二分类 macro 均值）
    // ============================================================

    public static double precisionScore(double[] yTrue, double[] yPred) {
        return precisionScore(yTrue, yPred, "macro");
    }

    public static double precisionScore(double[] yTrue, double[] yPred, String average) {
        Set<Double> classes = uniqueClasses(yTrue);
        double sum = 0;
        for (double c : classes) {
            int tp = 0, fp = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yPred[i] == c) { if (yTrue[i] == c) tp++; else fp++; }
            }
            int denom = tp + fp;
            sum += denom == 0 ? 0 : (double) tp / denom;
        }
        return "macro".equals(average) ? sum / classes.size() : sum;
    }

    public static double recallScore(double[] yTrue, double[] yPred) {
        return recallScore(yTrue, yPred, "macro");
    }

    public static double recallScore(double[] yTrue, double[] yPred, String average) {
        Set<Double> classes = uniqueClasses(yTrue);
        double sum = 0;
        for (double c : classes) {
            int tp = 0, fn = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == c) { if (yPred[i] == c) tp++; else fn++; }
            }
            int denom = tp + fn;
            sum += denom == 0 ? 0 : (double) tp / denom;
        }
        return "macro".equals(average) ? sum / classes.size() : sum;
    }

    public static double f1Score(double[] yTrue, double[] yPred) {
        return f1Score(yTrue, yPred, "macro");
    }

    public static double f1Score(double[] yTrue, double[] yPred, String average) {
        Set<Double> classes = uniqueClasses(yTrue);
        double sum = 0;
        for (double c : classes) {
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == c && yPred[i] == c) tp++;
                else if (yTrue[i] != c && yPred[i] == c) fp++;
                else if (yTrue[i] == c && yPred[i] != c) fn++;
            }
            double prec = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
            double rec  = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
            sum += (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
        }
        return "macro".equals(average) ? sum / classes.size() : sum;
    }

    // ============================================================
    // ROC AUC（二分类）
    // ============================================================

    public static double rocAucScore(double[] yTrue, double[] yScore) {
        int n = yTrue.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(yScore[b], yScore[a]));

        double auc = 0; double tpPrev = 0, fpPrev = 0; double tp = 0, fp = 0;
        double pos = 0; for (double v : yTrue) if (v == 1) pos++;
        double neg = n - pos;

        for (int i : idx) {
            if (yTrue[i] == 1) tp++;
            else { auc += tp; fp++; }
        }
        return auc / (pos * neg);
    }

    // ============================================================
    // 混淆矩阵
    // ============================================================

    public static int[][] confusionMatrix(double[] yTrue, double[] yPred) {
        List<Double> classList = new ArrayList<>(uniqueClasses(yTrue));
        Collections.sort(classList);
        int K = classList.size();
        int[][] cm = new int[K][K];
        for (int i = 0; i < yTrue.length; i++) {
            int ti = classList.indexOf(yTrue[i]);
            int pi = classList.indexOf(yPred[i]);
            if (ti >= 0 && pi >= 0) cm[ti][pi]++;
        }
        return cm;
    }

    /** 打印分类报告（类似 sklearn classification_report） */
    public static String classificationReport(double[] yTrue, double[] yPred) {
        List<Double> classList = new ArrayList<>(uniqueClasses(yTrue));
        Collections.sort(classList);
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-12s %9s %9s %9s %9s%n", "", "precision", "recall", "f1-score", "support"));
        sb.append(String.format("%-12s %9s %9s %9s %9s%n", "", "---------", "------", "--------", "-------"));

        double macroP = 0, macroR = 0, macroF1 = 0;
        for (double c : classList) {
            int tp = 0, fp = 0, fn = 0, support = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == c) support++;
                if (yTrue[i] == c && yPred[i] == c) tp++;
                else if (yTrue[i] != c && yPred[i] == c) fp++;
                else if (yTrue[i] == c && yPred[i] != c) fn++;
            }
            double prec = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
            double rec  = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
            double f1   = (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
            macroP += prec; macroR += rec; macroF1 += f1;
            sb.append(String.format("%-12s %9.2f %9.2f %9.2f %9d%n", (int) c, prec, rec, f1, support));
        }
        int K = classList.size();
        sb.append(String.format("%-12s %9.2f %9.2f %9.2f %9d%n",
            "macro avg", macroP / K, macroR / K, macroF1 / K, yTrue.length));
        sb.append(String.format("%-12s %9.2f%n", "accuracy", accuracyScore(yTrue, yPred)));
        return sb.toString();
    }

    // ============================================================
    // Jaccard / Cohen-Kappa / Hamming / Hinge / Log Loss
    // ============================================================

    public static double jaccardScore(double[] yTrue, double[] yPred) {
        int inter = 0, union = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 || yPred[i] == 1) union++;
            if (yTrue[i] == 1 && yPred[i] == 1) inter++;
        }
        return union == 0 ? 1.0 : (double) inter / union;
    }

    public static double cohenKappaScore(double[] y1, double[] y2) {
        double po = accuracyScore(y1, y2);
        Set<Double> cls = uniqueClasses(y1);
        double pe = 0;
        for (double c : cls) {
            long n1 = 0, n2 = 0; for (double v : y1) if (v == c) n1++; for (double v : y2) if (v == c) n2++;
            pe += ((double) n1 / y1.length) * ((double) n2 / y2.length);
        }
        return (po - pe) / (1 - pe + 1e-15);
    }

    public static double hammingLoss(double[] yTrue, double[] yPred) {
        int wrong = 0; for (int i = 0; i < yTrue.length; i++) if (yTrue[i] != yPred[i]) wrong++;
        return (double) wrong / yTrue.length;
    }

    public static double zeroOneLoss(double[] yTrue, double[] yPred) {
        return hammingLoss(yTrue, yPred);
    }

    public static double logLoss(double[] yTrue, double[] yProba) {
        double loss = 0;
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, yProba[i]));
            loss += -yTrue[i] * Math.log(p) - (1 - yTrue[i]) * Math.log(1 - p);
        }
        return loss / yTrue.length;
    }

    public static double brierScoreLoss(double[] yTrue, double[] yProba) {
        double s = 0; for (int i = 0; i < yTrue.length; i++) s += Math.pow(yProba[i] - yTrue[i], 2);
        return s / yTrue.length;
    }

    public static double hingeLoss(double[] yTrue, double[] decision) {
        double s = 0;
        for (int i = 0; i < yTrue.length; i++) {
            double y = yTrue[i] == 1 ? 1 : -1;
            s += Math.max(0, 1 - y * decision[i]);
        }
        return s / yTrue.length;
    }

    public static double averagePrecisionScore(double[] yTrue, double[] yScore) {
        int n = yTrue.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(yScore[b], yScore[a]));
        double ap = 0, tp = 0;
        for (int rank = 0; rank < n; rank++) {
            if (yTrue[idx[rank]] == 1) {
                tp++;
                ap += tp / (rank + 1);
            }
        }
        long pos = 0; for (double v : yTrue) if (v == 1) pos++;
        return pos == 0 ? 0 : ap / pos;
    }

    // ============================================================
    // ROC curve / precision-recall curve
    // ============================================================

    /** Returns [fpr, tpr, thresholds] */
    public static double[][] rocCurve(double[] yTrue, double[] yScore) {
        int n = yTrue.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(yScore[b], yScore[a]));

        long pos = 0; for (double v : yTrue) if (v == 1) pos++;
        long neg = n - pos;

        List<Double> fprs = new ArrayList<>(), tprs = new ArrayList<>(), threshs = new ArrayList<>();
        fprs.add(0.0); tprs.add(0.0); threshs.add(yScore[idx[0]] + 1);
        double tp = 0, fp = 0;
        for (int i : idx) {
            if (yTrue[i] == 1) tp++; else fp++;
            fprs.add(fp / neg);
            tprs.add(tp / pos);
            threshs.add(yScore[i]);
        }
        double[] fprArr = fprs.stream().mapToDouble(Double::doubleValue).toArray();
        double[] tprArr = tprs.stream().mapToDouble(Double::doubleValue).toArray();
        double[] thrArr = threshs.stream().mapToDouble(Double::doubleValue).toArray();
        return new double[][]{fprArr, tprArr, thrArr};
    }

    /** Returns [precision, recall, thresholds] */
    public static double[][] precisionRecallCurve(double[] yTrue, double[] yScore) {
        int n = yTrue.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(yScore[b], yScore[a]));

        List<Double> precs = new ArrayList<>(), recs = new ArrayList<>(), threshs = new ArrayList<>();
        double tp = 0, fp = 0;
        long pos = 0; for (double v : yTrue) if (v == 1) pos++;
        for (int i = 0; i < n; i++) {
            int ii = idx[i];
            if (yTrue[ii] == 1) tp++; else fp++;
            double prec = tp / (tp + fp);
            double rec  = tp / (pos + 1e-15);
            precs.add(prec); recs.add(rec); threshs.add(yScore[ii]);
        }
        precs.add(1.0); recs.add(0.0);
        return new double[][]{
            precs.stream().mapToDouble(Double::doubleValue).toArray(),
            recs.stream().mapToDouble(Double::doubleValue).toArray(),
            threshs.stream().mapToDouble(Double::doubleValue).toArray()
        };
    }

    // ============================================================
    // Clustering metrics
    // ============================================================

    public static double adjustedRandScore(double[] labelsTrue, double[] labelsPred) {
        int n = labelsTrue.length;
        Map<String, Integer> pairCounts = new HashMap<>();
        int[] trueCountMap = new int[n], predCountMap = new int[n];
        // Simplified implementation using contingency table
        Map<Double, Integer> trueIdx = new LinkedHashMap<>(), predIdx = new LinkedHashMap<>();
        for (double v : labelsTrue) trueIdx.putIfAbsent(v, trueIdx.size());
        for (double v : labelsPred) predIdx.putIfAbsent(v, predIdx.size());
        int R = trueIdx.size(), C = predIdx.size();
        int[][] cont = new int[R][C];
        for (int i = 0; i < n; i++) cont[trueIdx.get(labelsTrue[i])][predIdx.get(labelsPred[i])]++;
        double sumComb = 0, sumRow = 0, sumCol = 0;
        for (int i = 0; i < R; i++) {
            int rowSum = 0; for (int j = 0; j < C; j++) { rowSum += cont[i][j]; sumComb += comb2(cont[i][j]); }
            sumRow += comb2(rowSum);
        }
        for (int j = 0; j < C; j++) { int colSum = 0; for (int i = 0; i < R; i++) colSum += cont[i][j]; sumCol += comb2(colSum); }
        double totalComb = comb2(n);
        double expected = sumRow * sumCol / (totalComb + 1e-15);
        double maxVal = (sumRow + sumCol) / 2 - expected;
        return maxVal == 0 ? 0 : (sumComb - expected) / maxVal;
    }

    private static double comb2(int n) { return n * (n - 1) / 2.0; }

    private static Set<Double> uniqueClasses(double[] y) {
        TreeSet<Double> s = new TreeSet<>(); for (double v : y) s.add(v); return s;
    }
}

