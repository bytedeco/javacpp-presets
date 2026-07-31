/*
 * Simple expression transforms over row maps (column copy / rename / arithmetic).
 */
package org.bytedeco.pytorch.feature.transform;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/** Row-wise expression transform. */
public final class ExpressionTransform implements FeatureTransform {

    public enum Op {
        COPY, RENAME, ADD, SUB, MUL, DIV, LOG1P, ABS, CAST_DOUBLE, CAST_LONG, CONST
    }

    public static final class Step {
        public final Op op;
        public final String input;
        public final String input2;
        public final String output;
        public final double constant;

        public Step(Op op, String input, String input2, String output, double constant) {
            this.op = op;
            this.input = input;
            this.input2 = input2;
            this.output = output;
            this.constant = constant;
        }
    }

    private final String name;
    private final List<Step> steps;
    private final Function<Map<String, Object>, Map<String, Object>> custom;

    private ExpressionTransform(String name, List<Step> steps,
                                Function<Map<String, Object>, Map<String, Object>> custom) {
        this.name = name != null ? name : "expr";
        this.steps = List.copyOf(steps);
        this.custom = custom;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static ExpressionTransform custom(String name,
                                             Function<Map<String, Object>, Map<String, Object>> fn) {
        return new ExpressionTransform(name, List.of(), Objects.requireNonNull(fn, "fn"));
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public List<Map<String, Object>> apply(List<Map<String, Object>> rows) {
        if (rows == null) return List.of();
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            Map<String, Object> next = new LinkedHashMap<>(row);
            if (custom != null) {
                Map<String, Object> c = custom.apply(next);
                if (c != null) next.putAll(c);
            }
            for (Step step : steps) {
                applyStep(next, step);
            }
            out.add(next);
        }
        return out;
    }

    private static void applyStep(Map<String, Object> row, Step step) {
        switch (step.op) {
            case COPY:
            case RENAME: {
                Object v = row.get(step.input);
                row.put(step.output, v);
                if (step.op == Op.RENAME) row.remove(step.input);
                break;
            }
            case CONST:
                row.put(step.output, step.constant);
                break;
            case ADD:
                row.put(step.output, num(row.get(step.input)) + num(row.get(step.input2)));
                break;
            case SUB:
                row.put(step.output, num(row.get(step.input)) - num(row.get(step.input2)));
                break;
            case MUL:
                row.put(step.output, num(row.get(step.input)) * num(row.get(step.input2)));
                break;
            case DIV: {
                double d = num(row.get(step.input2));
                row.put(step.output, d == 0.0 ? Double.NaN : num(row.get(step.input)) / d);
                break;
            }
            case LOG1P:
                row.put(step.output, Math.log1p(Math.max(0.0, num(row.get(step.input)))));
                break;
            case ABS:
                row.put(step.output, Math.abs(num(row.get(step.input))));
                break;
            case CAST_DOUBLE:
                row.put(step.output, num(row.get(step.input)));
                break;
            case CAST_LONG:
                row.put(step.output, (long) num(row.get(step.input)));
                break;
            default:
                break;
        }
    }

    private static double num(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof Boolean) return ((Boolean) v) ? 1.0 : 0.0;
        if (v instanceof String) {
            try {
                return Double.parseDouble((String) v);
            } catch (NumberFormatException e) {
                return 0.0;
            }
        }
        return 0.0;
    }

    public static final class Builder {
        private String name = "expr";
        private final List<Step> steps = new ArrayList<>();

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder copy(String input, String output) {
            steps.add(new Step(Op.COPY, input, null, output, 0));
            return this;
        }

        public Builder rename(String input, String output) {
            steps.add(new Step(Op.RENAME, input, null, output, 0));
            return this;
        }

        public Builder add(String a, String b, String output) {
            steps.add(new Step(Op.ADD, a, b, output, 0));
            return this;
        }

        public Builder sub(String a, String b, String output) {
            steps.add(new Step(Op.SUB, a, b, output, 0));
            return this;
        }

        public Builder mul(String a, String b, String output) {
            steps.add(new Step(Op.MUL, a, b, output, 0));
            return this;
        }

        public Builder div(String a, String b, String output) {
            steps.add(new Step(Op.DIV, a, b, output, 0));
            return this;
        }

        public Builder log1p(String input, String output) {
            steps.add(new Step(Op.LOG1P, input, null, output, 0));
            return this;
        }

        public Builder constant(String output, double value) {
            steps.add(new Step(Op.CONST, null, null, output, value));
            return this;
        }

        public ExpressionTransform build() {
            return new ExpressionTransform(name, steps, null);
        }
    }
}
