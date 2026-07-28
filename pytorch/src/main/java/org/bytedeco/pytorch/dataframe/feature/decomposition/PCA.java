package org.bytedeco.pytorch.dataframe.feature.decomposition;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.List;

/**
 * Principal Component Analysis (sklearn-compatible core).
 *
 * <p>Fits via sample covariance + Jacobi symmetric eigendecomposition (not identity stub).
 * Exposes {@code components_}, {@code explained_variance_}, {@code explained_variance_ratio_},
 * {@code mean_}, {@code singular_values_} (approx via sqrt((n-1)*var)).
 *
 * <p>{@code nComponents} may be:
 * <ul>
 *   <li>positive int — keep that many components</li>
 *   <li>if constructed with variance fraction via {@link #PCA(double, String...)} where
 *       value is in (0,1], keep enough components to reach that cumulative variance</li>
 * </ul>
 */
public class PCA extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int nComponents;
    /** If in (0,1], select components by cumulative variance ratio. */
    private Double varianceThreshold = null;
    private double[][] components;          // [k, d] rows = principal axes
    private double[] mean;
    private double[] explainedVariance;
    private double[] explainedVarianceRatio;
    private double[] singularValues;
    private double noiseVariance;           // average leftover variance
    private boolean whiten = false;

    public PCA(int nComponents, String... columns) {
        super(columns);
        this.nComponents = Math.max(1, nComponents);
    }

    /** Keep components explaining at least {@code varianceFraction} of total variance (e.g. 0.85). */
    public PCA(double varianceFraction, String... columns) {
        super(columns);
        if (varianceFraction <= 0 || varianceFraction > 1.0) {
            throw new IllegalArgumentException("variance fraction must be in (0, 1]");
        }
        this.varianceThreshold = varianceFraction;
        this.nComponents = Integer.MAX_VALUE; // resolved in fit
    }

    public PCA setWhiten(boolean whiten) {
        this.whiten = whiten;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        String[] cols = columns.toArray(new String[0]);
        double[][] raw = FeatureMatrices.fromDf(X, cols);
        int n = raw.length;
        int d = cols.length;
        if (n < 2) throw new IllegalStateException("PCA needs at least 2 samples");

        // NaN -> column mean first for stability
        mean = new double[d];
        int[] cnt = new int[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double v = raw[i][j];
                if (!Double.isNaN(v)) { mean[j] += v; cnt[j]++; }
            }
        }
        for (int j = 0; j < d; j++) mean[j] = cnt[j] == 0 ? 0 : mean[j] / cnt[j];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                if (Double.isNaN(raw[i][j])) raw[i][j] = mean[j];
            }
        }

        // recompute mean on filled data
        mean = DenseLinalg.mean(raw);
        double[][] centered = DenseLinalg.center(raw, mean);
        double[][] cov = DenseLinalg.covariance(centered, true);

        DenseLinalg.EigenResult eig = DenseLinalg.eighSymmetric(cov);
        double[] evals = eig.eigenvalues; // desc
        double[][] evecs = eig.vectors;   // rows

        // numerical floor on eigenvalues
        double totalVar = 0;
        for (int i = 0; i < d; i++) {
            if (evals[i] < 0 && evals[i] > -1e-12) evals[i] = 0;
            totalVar += Math.max(0, evals[i]);
        }
        if (totalVar <= 0) totalVar = 1e-15;

        int k;
        if (varianceThreshold != null) {
            double cum = 0;
            k = 1;
            for (int i = 0; i < d; i++) {
                cum += Math.max(0, evals[i]) / totalVar;
                k = i + 1;
                if (cum >= varianceThreshold) break;
            }
            nComponents = k;
        } else {
            k = Math.min(nComponents, Math.min(n, d));
            nComponents = k;
        }

        components = new double[k][d];
        explainedVariance = new double[k];
        explainedVarianceRatio = new double[k];
        singularValues = new double[k];
        for (int i = 0; i < k; i++) {
            System.arraycopy(evecs[i], 0, components[i], 0, d);
            explainedVariance[i] = Math.max(0, evals[i]);
            explainedVarianceRatio[i] = explainedVariance[i] / totalVar;
            // s_i = sqrt((n-1) * lambda_i)  (sklearn convention for full SVD link)
            singularValues[i] = Math.sqrt(Math.max(0, (n - 1) * explainedVariance[i]));
        }

        double kept = 0;
        for (double v : explainedVariance) kept += v;
        noiseVariance = d > k ? Math.max(0, (totalVar - kept) / (d - k)) : 0.0;

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        String[] cols = columns.toArray(new String[0]);
        double[][] raw = FeatureMatrices.fromDf(X, cols);
        int n = raw.length;
        int d = cols.length;
        int k = components.length;

        // project: (X - mean) @ components.T
        double[][] projected = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    double v = raw[i][j];
                    if (Double.isNaN(v)) v = mean[j];
                    s += (v - mean[j]) * components[c][j];
                }
                if (whiten) {
                    double scale = explainedVariance[c] > 1e-15
                        ? 1.0 / Math.sqrt(explainedVariance[c]) : 0.0;
                    s *= scale;
                }
                projected[i][c] = s;
            }
        }

        DataFrame result = X.copy();
        for (int c = 0; c < k; c++) {
            String name = FeatureMatrices.uniqueName(result, "PC" + (c + 1));
            result.addColumn(name, Column.DType.FLOAT64);
            Column col = result.column(name);
            while (col.size() < n) col.add(null);
            for (int i = 0; i < n; i++) col.set(i, projected[i][c]);
        }
        return result;
    }

    /** Return only PC columns as a new frame. */
    public DataFrame transformToFrame(DataFrame X) throws Exception {
        requireFitted();
        DataFrame full = transform(X);
        String[] names = new String[components.length];
        for (int i = 0; i < names.length; i++) {
            // find PC columns we just named — prefer exact PC{i+1} then unique variants
            String base = "PC" + (i + 1);
            if (full.hasColumn(base)) names[i] = base;
            else {
                // last added unique
                for (Column c : full.columns()) {
                    if (c.name().startsWith(base)) { names[i] = c.name(); break; }
                }
            }
        }
        return full.select(names);
    }

    public double[][] getComponents() { return components; }
    public double[] getMean() { return mean; }
    public double[] getExplainedVariance() { return explainedVariance; }
    public double[] getExplainedVarianceRatio() { return explainedVarianceRatio; }
    public double[] getSingularValues() { return singularValues; }
    public double getNoiseVariance() { return noiseVariance; }
    public int getNComponents() { return nComponents; }

    /** Cumulative explained variance ratio. */
    public double[] getCumulativeExplainedVarianceRatio() {
        if (explainedVarianceRatio == null) return new double[0];
        double[] cum = new double[explainedVarianceRatio.length];
        double s = 0;
        for (int i = 0; i < cum.length; i++) {
            s += explainedVarianceRatio[i];
            cum[i] = s;
        }
        return cum;
    }
}
