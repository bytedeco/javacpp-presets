package org.bytedeco.pytorch.dataframe.ml.model_selection;

import java.util.*;

/** Leave-One-Out 交叉验证 */
public class LeaveOneOut {
    public List<KFold.Split> split(double[][] X, double[] y) {
        int n = X.length;
        List<KFold.Split> splits = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] test = {i};
            int[] train = new int[n - 1]; int t = 0;
            for (int j = 0; j < n; j++) if (j != i) train[t++] = j;
            splits.add(new KFold.Split(train, test));
        }
        return splits;
    }
    public int getNSplits(double[][] X) { return X.length; }
}

