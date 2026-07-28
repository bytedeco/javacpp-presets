package org.bytedeco.pytorch.dataframe.faiss;

import java.util.HashSet;
import java.util.Set;

/**
 * {@code faiss.IDSelectorArray} — select a fixed set of ids.
 */
public final class IDSelectorArray implements IDSelector {
    private final Set<Long> set;

    public IDSelectorArray(long[] ids) {
        this.set = new HashSet<>(ids == null ? 0 : ids.length * 2);
        if (ids != null) {
            for (long id : ids) set.add(id);
        }
    }

    public IDSelectorArray(int[] ids) {
        this.set = new HashSet<>(ids == null ? 0 : ids.length * 2);
        if (ids != null) {
            for (int id : ids) set.add((long) id);
        }
    }

    @Override
    public boolean is_member(long id) {
        return set.contains(id);
    }

    public long[] ids() {
        long[] out = new long[set.size()];
        int i = 0;
        for (Long v : set) out[i++] = v;
        return out;
    }

    public int size() {
        return set.size();
    }
}
