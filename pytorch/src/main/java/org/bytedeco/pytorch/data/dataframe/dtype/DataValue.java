package org.bytedeco.pytorch.data.dataframe.dtype;

import java.io.Serializable;

/**
 * Top interface for multimodal / structured cell values stored in {@link org.bytedeco.pytorch.data.dataframe.Column}.
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
