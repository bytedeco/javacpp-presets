package org.bytedeco.pytorch.data.dataframe;

import java.util.*;

/**
 * Rule-based lazy plan optimizer: filter merge + predicate pushdown past withColumn.
 */
final class LazyOptimizer {
    private LazyOptimizer() {}

    static List<LazyDataFrame.LazyOp> optimize(List<LazyDataFrame.LazyOp> plan) {
        if (plan == null || plan.isEmpty()) return List.of();
        List<LazyDataFrame.LazyOp> ops = new ArrayList<>(plan);

        // Pass 1: merge consecutive filters
        ops = mergeFilters(ops);

        // Pass 2: push filters before independent withColumns (bubble up)
        ops = pushFilters(ops);

        // Pass 3: merge filters again after push
        ops = mergeFilters(ops);

        return List.copyOf(ops);
    }

    private static List<LazyDataFrame.LazyOp> mergeFilters(List<LazyDataFrame.LazyOp> ops) {
        List<LazyDataFrame.LazyOp> out = new ArrayList<>();
        for (LazyDataFrame.LazyOp op : ops) {
            if (op instanceof LazyDataFrame.Filter f
                    && !out.isEmpty()
                    && out.get(out.size() - 1) instanceof LazyDataFrame.Filter prev) {
                Expression merged = prev.condition().and(f.condition());
                out.set(out.size() - 1, new LazyDataFrame.Filter(merged));
            } else {
                out.add(op);
            }
        }
        return out;
    }

    private static List<LazyDataFrame.LazyOp> pushFilters(List<LazyDataFrame.LazyOp> ops) {
        List<LazyDataFrame.LazyOp> out = new ArrayList<>(ops);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 1; i < out.size(); i++) {
                LazyDataFrame.LazyOp cur = out.get(i);
                LazyDataFrame.LazyOp prev = out.get(i - 1);
                // Don't push across CACHE barrier
                if (prev instanceof LazyDataFrame.Cache) continue;
                if (!(cur instanceof LazyDataFrame.Filter f)) continue;

                if (prev instanceof LazyDataFrame.WithColumn wc) {
                    Set<String> refs = f.referencedColumns();
                    // push if filter doesn't reference the new column
                    if (!refs.contains(wc.name())) {
                        // swap
                        out.set(i - 1, cur);
                        out.set(i, prev);
                        changed = true;
                    }
                } else if (prev instanceof LazyDataFrame.Rename ren) {
                    Set<String> refs = f.referencedColumns();
                    if (!refs.contains(ren.newName()) && !refs.contains(ren.oldName())) {
                        out.set(i - 1, cur);
                        out.set(i, prev);
                        changed = true;
                    }
                }
            }
        }
        return out;
    }
}
