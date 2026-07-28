package org.bytedeco.pytorch.dataframe.feature.selection;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Select K best features by score (sklearn SelectKBest).
 * Supervised scoring via {@link ScoreFunctions}: f_classif, f_regression, chi2.
 * Unsupervised fallback: column variance when no y is provided.
 */
public class SelectKBest extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int k;
    /** "f_classif" | "f_regression" | "chi2" | "variance" */
    private String scoreFunc = "f_classif";
    private String labelCol;
    private double[] externalY;
    private List<String> selectedColumns = new ArrayList<>();
    private final Map<String, Double> featureScores = new LinkedHashMap<>();

    public SelectKBest(int k, String... columns) {
        this(k, "f_classif", columns);
    }

    public SelectKBest(int k, String scoreFunc, String... columns) {
        super(columns);
        this.k = Math.max(1, k);
        this.scoreFunc = scoreFunc == null ? "f_classif" : scoreFunc.toLowerCase(Locale.ROOT);
    }

    public SelectKBest setLabelCol(String labelCol) {
        this.labelCol = labelCol;
        return this;
    }

    public SelectKBest setScoreFunc(String scoreFunc) {
        this.scoreFunc = scoreFunc == null ? "f_classif" : scoreFunc.toLowerCase(Locale.ROOT);
        return this;
    }

    /** sklearn-style fit with external y. */
    public SelectKBest fit(DataFrame X, double[] y) {
        this.externalY = y;
        fit(X);
        return this;
    }

    public SelectKBest fit(DataFrame X, String labelColumn) {
        this.labelCol = labelColumn;
        fit(X);
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        featureScores.clear();
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
            if (labelCol != null) columns.remove(labelCol);
        }
        String[] cols = columns.toArray(new String[0]);
        double[][] mat = FeatureMatrices.fromDf(X, cols);

        double[] y = resolveY(X, mat.length);
        double[] scores;
        if (y != null) {
            scores = switch (normalizeScore(scoreFunc)) {
                case "f_regression" -> ScoreFunctions.fRegression(mat, y);
                case "chi2" -> ScoreFunctions.chi2(mat, y);
                default -> ScoreFunctions.fClassif(mat, y);
            };
        } else {
            // unsupervised variance
            scores = new double[cols.length];
            for (int j = 0; j < cols.length; j++) {
                double mean = 0; int c = 0;
                for (int i = 0; i < mat.length; i++) {
                    if (!Double.isNaN(mat[i][j])) { mean += mat[i][j]; c++; }
                }
                mean = c == 0 ? 0 : mean / c;
                double var = 0;
                for (int i = 0; i < mat.length; i++) {
                    if (!Double.isNaN(mat[i][j])) {
                        double d = mat[i][j] - mean;
                        var += d * d;
                    }
                }
                scores[j] = c < 2 ? 0 : var / c;
            }
        }

        List<int[]> ranked = new ArrayList<>();
        for (int j = 0; j < cols.length; j++) {
            featureScores.put(cols[j], scores[j]);
            ranked.add(new int[]{j});
        }
        ranked.sort(Comparator.comparingDouble((int[] a) -> scores[a[0]]).reversed());

        int take = Math.min(k, cols.length);
        selectedColumns = new ArrayList<>();
        for (int i = 0; i < take; i++) {
            selectedColumns.add(cols[ranked.get(i)[0]]);
        }
        fitted = true;
        return this;
    }

    private double[] resolveY(DataFrame X, int n) {
        if (externalY != null) {
            if (externalY.length != n) {
                throw new IllegalArgumentException("y length must equal rowCount");
            }
            return externalY;
        }
        if (labelCol != null && X.hasColumn(labelCol)) {
            return FeatureMatrices.columnAsDoubles(X, labelCol);
        }
        return null;
    }

    private static String normalizeScore(String s) {
        if (s == null) return "f_classif";
        return switch (s) {
            case "f_score", "fscore", "f_classif", "fclassif" -> "f_classif";
            case "f_regression", "fregression" -> "f_regression";
            case "chi2", "chi_2" -> "chi2";
            case "variance", "var" -> "variance";
            default -> s;
        };
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        return X.select(selectedColumns.toArray(new String[0]));
    }

    public List<String> getSelectedColumns() { return selectedColumns; }
    public Map<String, Double> getScores() { return featureScores; }
    public int getK() { return k; }
}
