package org.bytedeco.pytorch.data.dataframe.feature.preprocessing;

import org.bytedeco.pytorch.data.dataframe.DataValues;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.feature.BaseTransformer;

import java.util.*;

/**
 * 样条变换器（对应 sklearn SplineTransformer）
 * 将数值特征扩展为 B 样条基函数特征
 */
public class SplineTransformer extends BaseTransformer {
    private int nKnots;
    private int degree;
    private Map<String, double[]> knots = new LinkedHashMap<>();  // per-column interior knots

    public SplineTransformer(String... columns) { this(5, 3, columns); }
    public SplineTransformer(int nKnots, int degree, String... columns) {
        super(columns); this.nKnots = nKnots; this.degree = degree;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        for (String col : columns) {
            List<Object> data = X.column(col).data();
            double[] vals = data.stream().filter(v -> v != null)
                .mapToDouble(v -> DataValues.asDouble(v)).sorted().toArray();
            if (vals.length == 0) { knots.put(col, new double[]{0, 1}); continue; }
            double min = vals[0], max = vals[vals.length - 1];
            double[] k = new double[nKnots + 2];
            for (int i = 0; i < nKnots + 2; i++) k[i] = min + (max - min) * i / (nKnots + 1);
            knots.put(col, k);
        }
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        if (!fitted) throw new IllegalStateException("SplineTransformer not fitted");
        DataFrame result = X.copy();
        for (String col : columns) {
            double[] k = knots.get(col);
            int nBases = k.length - 1 + degree; // approximate number of bases
            List<Object> data = X.column(col).data();
            for (int b = 0; b < nBases; b++) {
                final int bFinal = b;
                List<Double> basis = new ArrayList<>();
                for (Object v : data) {
                    double x = v == null ? 0.0 : DataValues.asDouble(v);
                    basis.add(bSplineBasis(x, bFinal, degree, k));
                }
                result = result.withColumnForDouble(col + "_spline_" + b, basis);
            }
        }
        return result;
    }

    /** Recursive B-spline basis function */
    private double bSplineBasis(double x, int i, int p, double[] t) {
        if (p == 0) return (x >= t[Math.min(i, t.length-1)] && x < t[Math.min(i+1, t.length-1)]) ? 1.0 : 0.0;
        double left = 0, right = 0;
        if (i < t.length - 1 && t[i + p] != t[i])
            left = (x - t[i]) / (t[i + p] - t[i]) * bSplineBasis(x, i, p - 1, t);
        if (i + 1 < t.length - 1 && t[i + p + 1] != t[i + 1])
            right = (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]) * bSplineBasis(x, i + 1, p - 1, t);
        return left + right;
    }
}

