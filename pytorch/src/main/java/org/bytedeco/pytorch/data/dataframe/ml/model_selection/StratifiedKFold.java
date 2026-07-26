package org.bytedeco.pytorch.data.dataframe.ml.model_selection;
import java.util.*;

/**
 * 分层 K 折（Stratified K-Fold）：保持每折中类别比例不变
 */
public class StratifiedKFold {
    private final int nSplits; private final boolean shuffle; private final Long randomState;

    public StratifiedKFold(int nSplits, boolean shuffle, Long randomState) {
        this.nSplits = nSplits; this.shuffle = shuffle; this.randomState = randomState;
    }
    public StratifiedKFold(int nSplits) { this(nSplits, false, null); }

    public List<KFold.Split> split(double[][] X, double[] y) {
        // Group indices by class
        Map<Double, List<Integer>> byClass = new LinkedHashMap<>();
        for (int i = 0; i < y.length; i++) byClass.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);

        Random rng = randomState == null ? new Random() : new Random(randomState);

        // Assign fold membership per class
        List<List<Integer>> folds = new ArrayList<>();
        for (int k = 0; k < nSplits; k++) folds.add(new ArrayList<>());

        for (Map.Entry<Double, List<Integer>> e : byClass.entrySet()) {
            List<Integer> classIdx = new ArrayList<>(e.getValue());
            if (shuffle) Collections.shuffle(classIdx, rng);
            for (int i = 0; i < classIdx.size(); i++) folds.get(i % nSplits).add(classIdx.get(i));
        }

        List<KFold.Split> splits = new ArrayList<>();
        for (int k = 0; k < nSplits; k++) {
            Set<Integer> testSet = new HashSet<>(folds.get(k));
            int[] test  = folds.get(k).stream().mapToInt(Integer::intValue).toArray();
            int[] train = new int[y.length - test.length]; int t = 0;
            for (int i = 0; i < y.length; i++) if (!testSet.contains(i)) train[t++] = i;
            splits.add(new KFold.Split(train, test));
        }
        return splits;
    }

    public int getNSplits() { return nSplits; }
}

