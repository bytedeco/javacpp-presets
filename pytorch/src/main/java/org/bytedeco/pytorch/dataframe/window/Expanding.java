package org.bytedeco.pytorch.dataframe.window;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.util.ArrayList;
import java.util.List;

/**
 * Pandas-style expanding window: {@code df.expanding().mean("x")}.
 */
public final class Expanding {
    private final DataFrame df;
    private final int minPeriods;

    public Expanding(DataFrame df) {
        this(df, 1);
    }

    public Expanding(DataFrame df, int minPeriods) {
        if (minPeriods <= 0) throw new IllegalArgumentException("minPeriods must be > 0");
        this.df = df;
        this.minPeriods = minPeriods;
    }

    public DataFrame mean(String column) { return reduce(column, column + "_expanding_mean", Op.MEAN); }
    public DataFrame sum(String column)  { return reduce(column, column + "_expanding_sum", Op.SUM); }
    public DataFrame min(String column)  { return reduce(column, column + "_expanding_min", Op.MIN); }
    public DataFrame max(String column)  { return reduce(column, column + "_expanding_max", Op.MAX); }
    public DataFrame std(String column)  { return reduce(column, column + "_expanding_std", Op.STD); }
    public DataFrame count(String column){ return reduce(column, column + "_expanding_count", Op.COUNT); }

    private enum Op { MEAN, SUM, MIN, MAX, STD, COUNT }

    private DataFrame reduce(String column, String outName, Op op) {
        List<Double> vals = numeric(column);
        DataFrame result = DataFrame.create();
        result.addColumn(outName, Column.DType.FLOAT64);

        double sum = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY, ss = 0;
        int count = 0;
        for (int i = 0; i < vals.size(); i++) {
            double v = vals.get(i);
            if (!Double.isNaN(v)) {
                count++;
                sum += v;
                min = Math.min(min, v);
                max = Math.max(max, v);
                ss += v * v;
            }
            double out;
            if (count < minPeriods) {
                out = Double.NaN;
            } else {
                out = switch (op) {
                    case SUM -> sum;
                    case MEAN -> sum / count;
                    case MIN -> min;
                    case MAX -> max;
                    case COUNT -> count;
                    case STD -> {
                        if (count < 2) yield Double.NaN;
                        double mean = sum / count;
                        double var = (ss - count * mean * mean) / (count - 1);
                        yield Math.sqrt(Math.max(0, var));
                    }
                };
            }
            result.addRow(out);
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
