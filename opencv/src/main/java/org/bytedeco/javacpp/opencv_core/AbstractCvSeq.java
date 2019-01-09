package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateSeq;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractCvSeq extends CvArr {
    public AbstractCvSeq(Pointer p) { super(p); }

    public static CvSeq create(int seq_flags, int header_size, int elem_size, CvMemStorage storage) {
        return cvCreateSeq(seq_flags, header_size, elem_size, storage);
    }
}
