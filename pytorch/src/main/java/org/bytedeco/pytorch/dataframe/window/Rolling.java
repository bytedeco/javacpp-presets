package org.bytedeco.pytorch.dataframe.window;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.util.ArrayList;
import java.util.List;

/**
 * Pandas-style fixed-window rolling aggregations: {@code df.rolling(window).mean("x")}.
 */
public final class Rolling {
    private final DataFrame df;
    private final int window;
    private final int minPeriods;

    public Rolling(DataFrame df, int window) {
        this(df, window, window);
    }

    public Rolling(DataFrame df, int window, int minPeriods) {
        if (window <= 0) throw new IllegalArgumentException("window must be > 0");
        if (minPeriods <= 0) throw new IllegalArgumentException("minPeriods must be > 0");
        this.df = df;
        this.window = window;
        this.minPeriods = minPeriods;
    }

    public DataFrame mean(String column) { return reduce(column, "rolling_mean", Op.MEAN); }
    public DataFrame sum(String column)  { return reduce(column, "rolling_sum", Op.SUM); }
    public DataFrame min(String column)  { return reduce(column, "rolling_min", Op.MIN); }
    public DataFrame max(String column)  { return reduce(column, "rolling_max", Op.MAX); }
    public DataFrame std(String column)  { return reduce(column, "rolling_std", Op.STD); }
    public DataFrame var(String column)  { return reduce(column, "rolling_var", Op.VAR); }
    public DataFrame count(String column){ return reduce(column, "rolling_count", Op.COUNT); }

    private enum Op { MEAN, SUM, MIN, MAX, STD, VAR, COUNT }

    private DataFrame reduce(String column, String outName, Op op) {
        List<Double> vals = numeric(column);
        DataFrame result = DataFrame.create();
        result.addColumn(outName, Column.DType.FLOAT64);
        for (int i = 0; i < vals.size(); i++) {
            int start = Math.max(0, i - window + 1);
            List<Double> win = new ArrayList<>();
            for (int j = start; j <= i; j++) {
                double v = vals.get(j);
                if (!Double.isNaN(v)) win.add(v);
            }
            double out;
            if (win.size() < minPeriods) {
                out = Double.NaN;
            } else {
                out = switch (op) {
                    case SUM -> win.stream().mapToDouble(d -> d).sum();
                    case MEAN -> win.stream().mapToDouble(d -> d).average().orElse(Double.NaN);
                    case MIN -> win.stream().mapToDouble(d -> d).min().orElse(Double.NaN);
                    case MAX -> win.stream().mapToDouble(d -> d).max().orElse(Double.NaN);
                    case COUNT -> win.size();
                    case VAR -> variance(win, true);
                    case STD -> Math.sqrt(variance(win, true));
                };
            }
            result.addRow(out);
        }
        return result;
    }

    private static double variance(List<Double> win, boolean sample) {
        int n = win.size();
        if (n == 0 || (sample && n < 2)) return Double.NaN;
        double mean = win.stream().mapToDouble(d -> d).average().orElse(0);
        double ss = 0;
        for (double v : win) ss += (v - mean) * (v - mean);
        return ss / (sample ? (n - 1) : n);
    }

    private List<Double> numeric(String column) {
        Column col = df.column(column);
        List<Double> out = new ArrayList<>(col.size());
        for (int i = 0; i < col.size(); i++) out.add(DataValues.asDouble(col.get(i)));
        return out;
    }
}
