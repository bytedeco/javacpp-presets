package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvSet extends CvSeq {
    public AbstractCvSet(Pointer p) { super(p); }

    public static CvSet create(int set_flags, int header_size, int elem_size,
            CvMemStorage storage) {
        return cvCreateSet(set_flags, header_size, elem_size, storage);
    }
}
