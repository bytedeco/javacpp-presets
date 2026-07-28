package org.bytedeco.pytorch.dataframe.feature.decomposition;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Linear Discriminant Analysis for supervised dimensionality reduction
 * (sklearn-compatible core: maximize between-class / within-class scatter).
 *
 * <p>Solves the generalized eigenproblem {@code Sb v = λ Sw v} via
 * {@code Sw^{-1} Sb} symmetric-ish route (Sw regularized) + Jacobi eigh.
 * At most {@code n_classes - 1} meaningful components.
 */
public class LDA extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nComponents;
    private String targetColumn;
    private double[][] components;   // [k, d]
    private double[] mean;           // overall mean
    private double[] explainedVarianceRatio;
    private final Map<Object, double[]> classMeans = new LinkedHashMap<>();
    private final Map<Object, Integer> classCounts = new LinkedHashMap<>();
    private double prioryReg = 1e-6;

    public LDA(int nComponents, String targetColumn, String... features) {
        super(features);
        this.nComponents = Math.max(1, nComponents);
        this.targetColumn = targetColumn;
    }

    public LDA setTargetColumn(String targetColumn) {
        this.targetColumn = targetColumn;
        return this;
    }

    public LDA setPrioryReg(double reg) {
        this.prioryReg = reg;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (targetColumn == null || !X.hasColumn(targetColumn)) {
            throw new IllegalStateException("LDA requires targetColumn in DataFrame");
        }
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
            columns.remove(targetColumn);
        }
        String[] feats = columns.toArray(new String[0]);
        double[][] raw = FeatureMatrices.fromDf(X, feats);
        int n = raw.length;
        int d = feats.length;

        // fill NaN with col means
        mean = DenseLinalg.mean(nanSafe(raw));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (Double.isNaN(raw[i][j])) raw[i][j] = mean[j];
        mean = DenseLinalg.mean(raw);

        // group by class
        classMeans.clear();
        classCounts.clear();
        Map<Object, List<Integer>> groups = new LinkedHashMap<>();
        Column tc = X.column(targetColumn);
        for (int i = 0; i < n; i++) {
            Object lab = DataValues.unwrap(tc.get(i));
            if (lab == null) lab = "null";
            groups.computeIfAbsent(lab, k -> new ArrayList<>()).add(i);
        }
        int C = groups.size();
        if (C < 2) throw new IllegalStateException("LDA needs at least 2 classes");

        // class means
        for (Map.Entry<Object, List<Integer>> e : groups.entrySet()) {
            List<Integer> idx = e.getValue();
            double[] cm = new double[d];
            for (int i : idx)
                for (int j = 0; j < d; j++) cm[j] += raw[i][j];
            for (int j = 0; j < d; j++) cm[j] /= idx.size();
            classMeans.put(e.getKey(), cm);
            classCounts.put(e.getKey(), idx.size());
        }

        // Within-class scatter Sw
        double[][] Sw = new double[d][d];
        for (Map.Entry<Object, List<Integer>> e : groups.entrySet()) {
            double[] cm = classMeans.get(e.getKey());
            for (int i : e.getValue()) {
                for (int j = 0; j < d; j++) {
                    double dj = raw[i][j] - cm[j];
                    for (int k = j; k < d; k++) {
                        Sw[j][k] += dj * (raw[i][k] - cm[k]);
                    }
                }
            }
        }
        for (int j = 0; j < d; j++) {
            for (int k = j; k < d; k++) Sw[k][j] = Sw[j][k];
            Sw[j][j] += prioryReg; // regularize
        }

        // Between-class scatter Sb
        double[][] Sb = new double[d][d];
        for (Map.Entry<Object, double[]> e : classMeans.entrySet()) {
            double[] cm = e.getValue();
            int nk = classCounts.get(e.getKey());
            double[] diff = new double[d];
            for (int j = 0; j < d; j++) diff[j] = cm[j] - mean[j];
            for (int j = 0; j < d; j++)
                for (int k = j; k < d; k++)
                    Sb[j][k] += nk * diff[j] * diff[k];
        }
        for (int j = 0; j < d; j++)
            for (int k = j; k < d; k++) Sb[k][j] = Sb[j][k];

        // Solve Sw^{-1} Sb via: for each column of Sb, solve Sw x = Sb_col, form M, then eigh(M)
        // Make M symmetric: use Sw^{-1/2} Sb Sw^{-1/2} approximately via solving.
        // Practical approach used by many pure-Java ports:
        //   M = solve(Sw, Sb) column-wise, then symmetrize and eigh.
        double[][] M = DenseLinalg.solveMulti(Sw, Sb);
        // symmetrize
        for (int i = 0; i < d; i++)
            for (int j = i + 1; j < d; j++) {
                double s = 0.5 * (M[i][j] + M[j][i]);
                M[i][j] = M[j][i] = s;
            }

        DenseLinalg.EigenResult eig = DenseLinalg.eighSymmetric(M);
        int maxK = Math.min(nComponents, C - 1);
        maxK = Math.min(maxK, d);
        nComponents = Math.max(1, maxK);

        components = new double[nComponents][d];
        explainedVarianceRatio = new double[nComponents];
        double sumPos = 0;
        for (int i = 0; i < d; i++) sumPos += Math.max(0, eig.eigenvalues[i]);
        if (sumPos <= 0) sumPos = 1e-15;
        for (int i = 0; i < nComponents; i++) {
            System.arraycopy(eig.vectors[i], 0, components[i], 0, d);
            explainedVarianceRatio[i] = Math.max(0, eig.eigenvalues[i]) / sumPos;
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        String[] feats = columns.toArray(new String[0]);
        double[][] raw = FeatureMatrices.fromDf(X, feats);
        int n = raw.length;
        int d = feats.length;
        int k = components.length;

        DataFrame result = X.copy();
        for (int c = 0; c < k; c++) {
            String name = FeatureMatrices.uniqueName(result, "LD" + (c + 1));
            result.addColumn(name, Column.DType.FLOAT64);
            Column col = result.column(name);
            while (col.size() < n) col.add(null);
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    double v = raw[i][j];
                    if (Double.isNaN(v)) v = mean[j];
                    // sklearn LDA transform typically centers by overall mean
                    s += (v - mean[j]) * components[c][j];
                }
                col.set(i, s);
            }
        }
        return result;
    }

    private static double[][] nanSafe(double[][] raw) {
        // temporary mean ignoring NaN for DenseLinalg.mean which doesn't handle NaN
        int n = raw.length, d = raw[0].length;
        double[][] copy = new double[n][d];
        double[] m = new double[d];
        int[] c = new int[d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                double v = raw[i][j];
                if (!Double.isNaN(v)) { m[j] += v; c[j]++; }
            }
        for (int j = 0; j < d; j++) m[j] = c[j] == 0 ? 0 : m[j] / c[j];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                copy[i][j] = Double.isNaN(raw[i][j]) ? m[j] : raw[i][j];
        return copy;
    }

    public double[][] getComponents() { return components; }
    public double[] getMean() { return mean; }
    public double[] getExplainedVarianceRatio() { return explainedVarianceRatio; }
    public Map<Object, double[]> getClassMeans() { return classMeans; }
    public int getNComponents() { return nComponents; }
}
