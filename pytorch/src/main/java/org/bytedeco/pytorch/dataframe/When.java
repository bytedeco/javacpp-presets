package org.bytedeco.pytorch.dataframe;

import java.util.ArrayList;
import java.util.List;

/**
 * Builder for {@code when(cond, value).when(...).otherwise(default)} expressions,
 * matching scala-polars / Polars ternary chains.
 *
 * <pre>
 *   Expression label = when(col("x").gt(0), "pos")
 *       .when(col("x").lt(0), "neg")
 *       .otherwise("zero");
 * </pre>
 */
public final class When {
    private final List<Expression> conditions = new ArrayList<>();
    private final List<Expression> values = new ArrayList<>();

    When() {}

    /** Add another when branch. */
    public When when(Expression condition, Object value) {
        conditions.add(condition);
        values.add(Expression.toExpr(value));
        return this;
    }

    /** Terminate the chain with a default value; returns a complete Expression. */
    public Expression otherwise(Object value) {
        return new Expression.WhenThenExpr(
            new ArrayList<>(conditions),
            new ArrayList<>(values),
            Expression.toExpr(value));
    }

    /**
     * Terminate without an otherwise — unmatched rows yield null.
     * Convenience when the default is null.
     */
    public Expression otherwiseNull() {
        return otherwise(null);
    }
}
