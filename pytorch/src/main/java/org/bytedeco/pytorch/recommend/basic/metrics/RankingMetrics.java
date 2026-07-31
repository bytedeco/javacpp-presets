/*
 * Ported from torchSa: torchrec/metrics/RankingMetrics.scala
 *
 * Ranking / CTR evaluation metrics on CPU from score/label arrays.
 * Includes a streaming Accumulator with primitive float buffers (no boxing)
 * so epoch-level micro-AUC does not cause progressive GC slowdown.
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.recommend.TensorHelpers;

import java.util.Arrays;
import java.util.Comparator;

public final class RankingMetrics {

    private RankingMetrics() {}

    /** Binary log-loss (mean). scores are probabilities in (0,1). */
    public static double logLoss(float[] scores, float[] labels) {
        return logLoss(scores, labels, 1e-7f, -1);
    }

    public static double logLoss(float[] scores, float[] labels, float eps, int n) {
        int m = n >= 0
                ? Math.min(n, Math.min(scores.length, labels.length))
                : Math.min(scores.length, labels.length);
        if (m == 0) return 0.0;
        double s = 0.0;
        for (int i = 0; i < m; i++) {
            double p = Math.min(1.0 - eps, Math.max(eps, scores[i]));
            double y = labels[i];
            s += -(y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p));
        }
        return s / m;
    }

    /** Accuracy at threshold 0.5. */
    public static double accuracy(float[] scores, float[] labels) {
        return accuracy(scores, labels, 0.5f, -1);
    }

    public static double accuracy(float[] scores, float[] labels, float thr, int n) {
        int m = n >= 0
                ? Math.min(n, Math.min(scores.length, labels.length))
                : Math.min(scores.length, labels.length);
        if (m == 0) return 0.0;
        int correct = 0;
        for (int i = 0; i < m; i++) {
            float pred = scores[i] >= thr ? 1.0f : 0.0f;
            if (pred == labels[i]) correct++;
        }
        return (double) correct / m;
    }

    /**
     * ROC-AUC via Mann–Whitney (average ranks for ties).
     * Returns 0.5 when only one class is present.
     */
    public static double auc(float[] scores, float[] labels) {
        return auc(scores, labels, -1);
    }

    public static double auc(float[] scores, float[] labels, int n) {
        int m = n >= 0
                ? Math.min(n, Math.min(scores.length, labels.length))
                : Math.min(scores.length, labels.length);
        if (m == 0) return 0.5;

        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(i -> scores[i]));

        long pos = 0L;
        long neg = 0L;
        double rankSumPos = 0.0;
        int i = 0;
        while (i < m) {
            int j = i;
            float v = scores[order[i]];
            while (j < m && scores[order[j]] == v) j++;
            double avgRank = (i + 1 + j) / 2.0; // 1-based ranks
            for (int k = i; k < j; k++) {
                if (labels[order[k]] > 0.5f) {
                    pos++;
                    rankSumPos += avgRank;
                } else {
                    neg++;
                }
            }
            i = j;
        }
        if (pos == 0 || neg == 0) return 0.5;
        return (rankSumPos - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /** Copy a float tensor (any device/shape) to a flat JVM float[]. */
    public static float[] toFloatArray(Tensor t) {
        Tensor cpu = t.detach().to(ScalarType.Float).cpu().contiguous();
        return TensorHelpers.toFloatArray(cpu);
    }

    /**
     * Copy a float tensor into an existing destination buffer starting at {@code destOff}.
     * Returns number of elements written.
     */
    public static int copyFloatArray(Tensor t, float[] dest, int destOff) {
        Tensor cpu = t.detach().to(ScalarType.Float).cpu().contiguous();
        float[] tmp = TensorHelpers.toFloatArray(cpu);
        int n = tmp.length;
        if (n <= 0) return 0;
        if (destOff + n > dest.length) {
            throw new IllegalArgumentException(
                    "copyFloatArray overflow: destOff=" + destOff + " n=" + n
                            + " dest.length=" + dest.length);
        }
        System.arraycopy(tmp, 0, dest, destOff, n);
        return n;
    }

    public static final class BatchMetrics {
        public final double logloss;
        public final double auc;
        public final double accuracy;
        public final int n;

        public BatchMetrics(double logloss, double auc, double accuracy, int n) {
            this.logloss = logloss;
            this.auc = auc;
            this.accuracy = accuracy;
            this.n = n;
        }
    }

    public static BatchMetrics fromLogits(Tensor logits, Tensor labels) {
        float[] s = toFloatArray(logits.detach().sigmoid().reshape(-1L));
        float[] y = toFloatArray(labels.reshape(-1L));
        int n = Math.min(s.length, y.length);
        return new BatchMetrics(logLoss(s, y, 1e-7f, n), auc(s, y, n), accuracy(s, y, 0.5f, n), n);
    }

    /**
     * Streaming accumulator for epoch-level micro-AUC / mean BCE / accuracy.
     * Primitive float buffers — no boxing, geometric growth, optional capacity hint.
     */
    public static final class Accumulator {
        private float[] scores;
        private float[] labels;
        private int size;
        private double lossSum;
        private int lossN;

        public Accumulator() {
            this(4096);
        }

        public Accumulator(int initialCapacity) {
            int cap = Math.max(64, initialCapacity);
            this.scores = new float[cap];
            this.labels = new float[cap];
            this.size = 0;
            this.lossSum = 0.0;
            this.lossN = 0;
        }

        public void ensureCapacity(int extra) {
            int need = size + Math.max(0, extra);
            if (need > scores.length) growTo(need);
        }

        private void growTo(int minCap) {
            int cap = Math.max(scores.length, 64);
            while (cap < minCap) {
                long next = (long) cap * 2L;
                if (next >= Integer.MAX_VALUE) {
                    cap = minCap;
                    break;
                }
                cap = (int) next;
            }
            if (cap < minCap) cap = minCap;
            float[] ns = new float[cap];
            float[] nl = new float[cap];
            if (size > 0) {
                System.arraycopy(scores, 0, ns, 0, size);
                System.arraycopy(labels, 0, nl, 0, size);
            }
            scores = ns;
            labels = nl;
        }

        private void ensureRoom(int extra) {
            int need = size + extra;
            if (need > scores.length) growTo(need);
        }

        /**
         * Append batch logits/labels and accumulate weighted BCE.
         * Sigmoid on device, then one CPU pull for scores and one for labels.
         */
        public void update(Tensor logits, Tensor labelTensor, double bceLossValue, int batchSize) {
            Tensor probT = logits.detach().sigmoid().reshape(-1L);
            Tensor labT = labelTensor.detach().reshape(-1L);
            int nS = (int) probT.__dispatch_numel();
            int nY = (int) labT.__dispatch_numel();
            int n = Math.min(nS, nY);
            if (n > 0) {
                ensureRoom(n);
                Tensor probN = nS == n ? probT : probT.narrow(0, 0, n);
                Tensor labN = nY == n ? labT : labT.narrow(0, 0, n);
                copyFloatArray(probN, scores, size);
                copyFloatArray(labN, labels, size);
                size += n;
            }
            lossSum += bceLossValue * batchSize;
            lossN += batchSize;
        }

        /** Track running BCE only — no score/label storage, O(1). */
        public void updateLossOnly(double bceLossValue, int batchSize) {
            lossSum += bceLossValue * batchSize;
            lossN += batchSize;
        }

        public int sampleCount() { return size; }
        public int lossCount() { return lossN; }

        public void clear() {
            size = 0;
            lossSum = 0.0;
            lossN = 0;
            // keep capacity for next epoch
        }

        /** (logloss, auc, accuracy, n) — O(n log n) for AUC, call once per epoch. */
        public double[] result() {
            int n = size;
            double logloss = lossN > 0 ? lossSum / lossN : logLoss(scores, labels, 1e-7f, n);
            if (n == 0) {
                return new double[]{logloss, 0.5, 0.0, 0.0};
            }
            return new double[]{logloss, auc(scores, labels, n), accuracy(scores, labels, 0.5f, n), n};
        }
    }
}
