package org.bytedeco.pytorch.dataframe.window;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.util.ArrayList;
import java.util.List;

/**
 * Exponentially weighted moments — pandas-style {@code df.ewm(alpha).mean("x")}.
 */
public final class Ewm {
    private final DataFrame df;
    private final double alpha;
    private final boolean adjust;

    public Ewm(DataFrame df, double alpha) {
        this(df, alpha, true);
    }

    public Ewm(DataFrame df, double alpha, boolean adjust) {
        if (alpha <= 0 || alpha > 1) throw new IllegalArgumentException("alpha must be in (0,1]");
        this.df = df;
        this.alpha = alpha;
        this.adjust = adjust;
    }

    /** Build from span: alpha = 2 / (span + 1). */
    public static Ewm fromSpan(DataFrame df, double span) {
        if (span < 1) throw new IllegalArgumentException("span must be >= 1");
        return new Ewm(df, 2.0 / (span + 1.0), true);
    }

    /** Build from center of mass: alpha = 1 / (1 + com). */
    public static Ewm fromCom(DataFrame df, double com) {
        if (com < 0) throw new IllegalArgumentException("com must be >= 0");
        return new Ewm(df, 1.0 / (1.0 + com), true);
    }

    public DataFrame mean(String column) {
        List<Double> vals = numeric(column);
        DataFrame result = DataFrame.create();
        result.addColumn(column + "_ewm_mean", Column.DType.FLOAT64);
        if (vals.isEmpty()) return result;

        if (adjust) {
            for (int i = 0; i < vals.size(); i++) {
                double num = 0, den = 0, w = 1;
                for (int k = i; k >= 0; k--) {
                    double v = vals.get(k);
                    if (!Double.isNaN(v)) {
                        num += w * v;
                        den += w;
                    }
                    w *= (1 - alpha);
                }
                result.addRow(den == 0 ? Double.NaN : num / den);
            }
        } else {
            double prev = Double.NaN;
            for (int i = 0; i < vals.size(); i++) {
                double v = vals.get(i);
                if (Double.isNaN(v)) {
                    result.addRow(Double.NaN);
                    continue;
                }
                prev = Double.isNaN(prev) ? v : alpha * v + (1 - alpha) * prev;
                result.addRow(prev);
            }
        }
        return result;
    }

    private List<Double> numeric(String column) {
        Column col = df.column(column);
        List<Double> out = new ArrayList<>(col.size());
        for (int i = 0; i < col.size(); i++) out.add(DataValues.asDouble(col.get(i)));
        return out;
    }
}
