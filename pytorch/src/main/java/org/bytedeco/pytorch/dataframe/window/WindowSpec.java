package org.bytedeco.pytorch.dataframe.window;

import org.bytedeco.pytorch.dataframe.Expression;

import java.util.*;

/**
 * Spark-style window specification: partitionBy / orderBy / frame.
 *
 * <pre>
 *   WindowSpec w = WindowSpec.partitionBy("dept")
 *       .orderBy(Functions.asc("salary"))
 *       .rowsBetween(-2, 0);
 * </pre>
 */
public final class WindowSpec {
    private final String[] partitionBy;
    private final Expression[] orderBy;   // SortKeyExpr or plain col expressions
    private final boolean[] orderDesc;    // parallel to orderBy; true = desc
    private final WindowFrame frame;

    private WindowSpec(String[] partitionBy, Expression[] orderBy, boolean[] orderDesc, WindowFrame frame) {
        this.partitionBy = partitionBy == null ? new String[0] : partitionBy.clone();
        this.orderBy = orderBy == null ? new Expression[0] : orderBy.clone();
        this.orderDesc = orderDesc == null ? new boolean[this.orderBy.length] : orderDesc.clone();
        this.frame = frame != null ? frame
            : (this.orderBy.length > 0 ? WindowFrame.defaultOrdered() : WindowFrame.wholePartition());
    }

    public static WindowSpec empty() {
        return new WindowSpec(null, null, null, null);
    }

    /**
     * Set partition columns: {@code window().partitionBy("dept")} or
     * {@code WindowSpec.empty().partitionBy("dept")}.
     */
    public WindowSpec partitionBy(String... cols) {
        return new WindowSpec(cols, this.orderBy, this.orderDesc, this.frame);
    }

    public WindowSpec orderBy(Expression... keys) {
        Expression[] exprs = new Expression[keys.length];
        boolean[] desc = new boolean[keys.length];
        for (int i = 0; i < keys.length; i++) {
            Expression k = keys[i];
            if (k != null && k.isSortKey()) {
                exprs[i] = k.sortChild();
                desc[i] = k.isSortDescending();
            } else {
                exprs[i] = k;
                desc[i] = false;
            }
        }
        return new WindowSpec(this.partitionBy, exprs, desc, this.frame);
    }

    public WindowSpec orderBy(String... cols) {
        Expression[] exprs = new Expression[cols.length];
        boolean[] desc = new boolean[cols.length];
        for (int i = 0; i < cols.length; i++) {
            exprs[i] = Expression.col(cols[i]);
            desc[i] = false;
        }
        return new WindowSpec(this.partitionBy, exprs, desc, this.frame);
    }

    public WindowSpec rowsBetween(long start, long end) {
        return new WindowSpec(partitionBy, orderBy, orderDesc, WindowFrame.rows(start, end));
    }

    public WindowSpec rangeBetween(long start, long end) {
        return new WindowSpec(partitionBy, orderBy, orderDesc, WindowFrame.range(start, end));
    }

    public WindowSpec frame(WindowFrame f) {
        return new WindowSpec(partitionBy, orderBy, orderDesc, f);
    }

    public String[] partitionBy() { return partitionBy.clone(); }
    public Expression[] orderBy() { return orderBy.clone(); }
    public boolean[] orderDesc() { return orderDesc.clone(); }
    public WindowFrame frame() { return frame; }

    public boolean hasOrder() { return orderBy.length > 0; }
    public boolean hasPartition() { return partitionBy.length > 0; }

    @Override
    public String toString() {
        return "Window(partitionBy=" + Arrays.toString(partitionBy)
            + ", orderBy=" + Arrays.toString(orderBy)
            + ", frame=" + frame + ")";
    }
}
