package org.bytedeco.pytorch.dataframe.feature.construction;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.List;

/**
 * Generate polynomial and interaction features (sklearn PolynomialFeatures).
 * Supports {@code interaction_only} and {@code include_bias}.
 */
public class PolynomialFeatures extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int degree = 2;
    private boolean includeBias = false;
    private boolean interactionOnly = false;
    /** Each term is a multiset of input column indices (exponents via counts). */
    private List<int[]> terms = new ArrayList<>();
    private List<String> featureNames = new ArrayList<>();

    public PolynomialFeatures(int degree, String... columns) {
        this(degree, false, false, columns);
    }

    public PolynomialFeatures(int degree, boolean includeBias, String... columns) {
        this(degree, includeBias, false, columns);
    }

    public PolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly, String... columns) {
        super(columns);
        this.degree = Math.max(1, degree);
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
    }

    public PolynomialFeatures setInteractionOnly(boolean interactionOnly) {
        this.interactionOnly = interactionOnly;
        return this;
    }

    public PolynomialFeatures setIncludeBias(boolean includeBias) {
        this.includeBias = includeBias;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        terms = new ArrayList<>();
        featureNames = new ArrayList<>();
        if (includeBias) {
            terms.add(new int[0]);
            featureNames.add("1");
        }
        int d = columns.size();
        // Generate all multi-indices of total degree 1..degree
        generateTerms(d, degree, new int[0]);
        fitted = true;
        return this;
    }

    private void generateTerms(int nFeatures, int maxDegree, int[] prefix) {
        int start = prefix.length == 0 ? 0 : prefix[prefix.length - 1];
        for (int f = start; f < nFeatures; f++) {
            int[] next = new int[prefix.length + 1];
            System.arraycopy(prefix, 0, next, 0, prefix.length);
            next[prefix.length] = f;
            int deg = next.length;
            if (deg > maxDegree) continue;
            if (acceptTerm(next)) {
                terms.add(next);
                featureNames.add(nameOf(next));
            }
            if (deg < maxDegree) generateTerms(nFeatures, maxDegree, next);
        }
    }

    private boolean acceptTerm(int[] term) {
        if (!interactionOnly) return true;
        // interaction_only: no pure powers (all indices distinct) and degree >= 2 for interactions,
        // but sklearn still includes degree-1 (raw features).
        if (term.length <= 1) return true;
        for (int i = 1; i < term.length; i++) {
            if (term[i] == term[i - 1]) return false; // repeated → pure power component
        }
        return true;
    }

    private String nameOf(int[] term) {
        if (term.length == 0) return "1";
        // count exponents
        int n = columns.size();
        int[] exp = new int[n];
        for (int idx : term) exp[idx]++;
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < n; j++) {
            if (exp[j] == 0) continue;
            if (sb.length() > 0) sb.append(' ');
            sb.append(columns.get(j));
            if (exp[j] > 1) sb.append('^').append(exp[j]);
        }
        return sb.toString().replace(' ', '_');
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        int n = X.rowCount();
        int dIn = columns.size();
        double[][] src = FeatureMatrices.fromDf(X, columns.toArray(new String[0]));
        int dOut = terms.size();
        double[][] out = new double[n][dOut];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < dOut; t++) {
                int[] term = terms.get(t);
                double v = 1.0;
                for (int idx : term) {
                    v *= src[i][idx];
                }
                out[i][t] = v;
            }
        }
        // append poly columns (keep originals)
        String[] names = new String[dOut];
        for (int t = 0; t < dOut; t++) {
            names[t] = "poly_" + featureNames.get(t);
        }
        return FeatureMatrices.appendColumns(X, names, out);
    }

    /** Return only polynomial feature matrix as a new DataFrame (no original cols). */
    public DataFrame transformToFrame(DataFrame X) throws Exception {
        requireFitted();
        int n = X.rowCount();
        double[][] src = FeatureMatrices.fromDf(X, columns.toArray(new String[0]));
        int dOut = terms.size();
        double[][] out = new double[n][dOut];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < dOut; t++) {
                double v = 1.0;
                for (int idx : terms.get(t)) v *= src[i][idx];
                out[i][t] = v;
            }
        }
        String[] names = new String[dOut];
        for (int t = 0; t < dOut; t++) names[t] = "poly_" + featureNames.get(t);
        return FeatureMatrices.toDf(out, names);
    }

    public List<String> getFeatureNames() { return featureNames; }
    public boolean isInteractionOnly() { return interactionOnly; }
    public int getDegree() { return degree; }
}
