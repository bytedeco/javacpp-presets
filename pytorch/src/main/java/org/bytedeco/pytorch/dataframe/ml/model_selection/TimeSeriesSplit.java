package org.bytedeco.pytorch.dataframe.ml.model_selection;

import java.util.*;

/** Time Series 分割（不打乱顺序） */
public class TimeSeriesSplit {
    private final int nSplits; private final int gap; private final Integer maxTrainSize;

    public TimeSeriesSplit(int nSplits) { this(nSplits, 0, null); }
    public TimeSeriesSplit(int nSplits, int gap, Integer maxTrainSize) {
        this.nSplits = nSplits; this.gap = gap; this.maxTrainSize = maxTrainSize;
    }

    public List<KFold.Split> split(double[][] X, double[] y) {
        int n = X.length;
        int testSize = n / (nSplits + 1);
        List<KFold.Split> splits = new ArrayList<>();
        for (int k = 0; k < nSplits; k++) {
            int trainEnd = (k + 1) * testSize;
            int testStart = trainEnd + gap;
            int testEnd = testStart + testSize;
            if (testEnd > n) testEnd = n;
            if (testStart >= testEnd) continue;
            int trainStart = maxTrainSize == null ? 0 : Math.max(0, trainEnd - maxTrainSize);
            int[] train = new int[trainEnd - trainStart];
            for (int i = trainStart; i < trainEnd; i++) train[i - trainStart] = i;
            int[] test = new int[testEnd - testStart];
            for (int i = testStart; i < testEnd; i++) test[i - testStart] = i;
            splits.add(new KFold.Split(train, test));
        }
        return splits;
    }
}

