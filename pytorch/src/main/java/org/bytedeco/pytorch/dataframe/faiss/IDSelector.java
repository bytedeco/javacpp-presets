package org.bytedeco.pytorch.dataframe.faiss;

/**
 * ID filter for {@link Index#remove_ids(IDSelector)} — mirrors {@code faiss.IDSelector}.
 */
public interface IDSelector {
    /** Whether the given id should be removed / selected. */
    boolean is_member(long id);
}
