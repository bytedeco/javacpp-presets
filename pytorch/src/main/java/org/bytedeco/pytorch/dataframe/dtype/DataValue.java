package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.Column;

import java.io.Serializable;

/**
 * Top interface for multimodal / structured cell values stored in {@link Column}.
 * Plain numeric/string cells remain unwrapped {@link Number}/{@link String} etc.
 */
public interface DataValue extends Serializable {
    /** Type tag e.g. "IMAGE", "AUDIO", "JSON". */
    String getDataType();

    /** Arrow / storage-compatible payload (Number, String, List, Map, byte[], …). */
    Object toArrowCompatible();

    boolean isValid();

    String getShortDesc();
}
